package main

import (
	"encoding/binary"
	"fmt"
	"image/color"
	"math"
	"os"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

const (
	screenWidth  = 800
	screenHeight = 600
	frameDelay   = 16 * time.Millisecond // Match the C program's 60 FPS

	// Camera movement speeds
	moveSpeed        = 0.5
	rotateSpeed      = 0.02
	mouseRotateSpeed = 0.003
)

type Camera struct {
	PosX, PosY, PosZ float32
	DirX, DirY, DirZ float32
	FOV              float32
	LastMouseX       int
	LastMouseY       int
	MouseDragging    bool
}

type Game struct {
	pixels     [screenWidth * screenHeight * 4]uint8
	img        *ebiten.Image
	lastUpdate time.Time
	camera     Camera
}

func PlotFPS(screen *ebiten.Image) {
	// Plot parameters
	const (
		samples     = 30                 // Number of samples to plot
		graphX      = 50                 // Left margin
		graphY      = screenHeight - 110 // Bottom margin
		graphWidth  = 180                // Width of graph
		graphHeight = 100                // Height of graph
	)

	// read binary data containing FPS data
	fpsData, err := os.ReadFile("average_fps.bin")
	if err != nil {
		fmt.Println("Error reading average_fps.bin:", err)
		return
	}
	if len(fpsData) < samples*4 {
		fmt.Println("average_fps.bin does not contain enough data")
		return
	}

	// find min and max
	minFPS := float32(1e6)
	maxFPS := float32(-1e6)
	fps := make([]float32, samples)

	for i := 0; i < samples; i++ {
		fps[i] = math.Float32frombits(binary.LittleEndian.Uint32(fpsData[i*4 : (i+1)*4]))
		if fps[i] < minFPS {
			minFPS = fps[i]
		}
		if fps[i] > maxFPS {
			maxFPS = fps[i]
		}
	}

	// Ensure we have reasonable min/max values
	if maxFPS <= minFPS {
		maxFPS = minFPS + 10
	}

	// Draw axis
	ebitenutil.DrawLine(screen, graphX, graphY, graphX+graphWidth, graphY, color.RGBA{100, 100, 100, 255})  // X axis
	ebitenutil.DrawLine(screen, graphX, graphY-graphHeight, graphX, graphY, color.RGBA{100, 100, 100, 255}) // Y axis

	// Calculate step size to fill the entire graph width
	stepWidth := float64(graphWidth) / float64(samples-1)

	// Draw FPS values
	for i := 0; i < len(fps)-1; i++ {
		x1 := float64(graphX) + float64(i)*stepWidth
		x2 := float64(graphX) + float64(i+1)*stepWidth

		// Calculate normalized heights
		y1 := float64(graphY - (fps[i]-minFPS)/(maxFPS-minFPS)*float32(graphHeight))
		y2 := float64(graphY - (fps[i+1]-minFPS)/(maxFPS-minFPS)*float32(graphHeight))

		// Color based on FPS value (green for high, yellow for medium, red for low)
		r := uint8(255 * (1 - (fps[i]-minFPS)/(maxFPS-minFPS)))
		g := uint8(255 * (fps[i] - minFPS) / (maxFPS - minFPS))
		ebitenutil.DrawLine(screen, x1, y1, x2, y2, color.RGBA{r, g, 0, 255})
	}

	// Draw min/max labels
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("%.1f", maxFPS), graphX-30, int(graphY-graphHeight))
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("%.1f", minFPS), graphX-30, int(graphY))
}

func (g *Game) UpdatePixels() error {
	// Open the file directly for binary reading
	data, err := os.ReadFile("output.bin")
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// The C program writes data by rows (y) then columns (x)
	// Need to remap to RGBA format for Ebiten
	for y := 0; y < screenHeight; y++ {
		for x := 0; x < screenWidth; x++ {
			srcPos := (y*screenWidth + x) * 4
			dstPos := (y*screenWidth + x) * 4

			// Check bounds to prevent potential panic
			if srcPos+3 < len(data) && dstPos+3 < len(g.pixels) {
				g.pixels[dstPos] = data[srcPos]     // R
				g.pixels[dstPos+1] = data[srcPos+1] // G
				g.pixels[dstPos+2] = data[srcPos+2] // B
				g.pixels[dstPos+3] = data[srcPos+3] // A
			}
		}
	}

	// Update the image
	if g.img == nil {
		g.img = ebiten.NewImage(screenWidth, screenHeight)
	}
	g.img.ReplacePixels(g.pixels[:])
	return nil
}

func (g *Game) writeCameraData() error {
	// Create a binary file to share camera data with the C program
	file, err := os.Create("camera.bin")
	if err != nil {
		return fmt.Errorf("failed to create camera file: %w", err)
	}
	defer file.Close()

	// Write camera data in binary format
	binary.Write(file, binary.LittleEndian, g.camera.PosX)
	binary.Write(file, binary.LittleEndian, g.camera.PosY)
	binary.Write(file, binary.LittleEndian, g.camera.PosZ)
	binary.Write(file, binary.LittleEndian, g.camera.DirX)
	binary.Write(file, binary.LittleEndian, g.camera.DirY)
	binary.Write(file, binary.LittleEndian, g.camera.DirZ)
	binary.Write(file, binary.LittleEndian, g.camera.FOV)

	return nil
}

func (g *Game) handleCameraMovement() {
	// Keyboard controls for position
	if ebiten.IsKeyPressed(ebiten.KeyW) || ebiten.IsKeyPressed(ebiten.KeyUp) {
		g.camera.PosZ += moveSpeed * g.camera.DirZ
		g.camera.PosX += moveSpeed * g.camera.DirX
		g.camera.PosY += moveSpeed * g.camera.DirY
	}
	if ebiten.IsKeyPressed(ebiten.KeyS) || ebiten.IsKeyPressed(ebiten.KeyDown) {
		g.camera.PosZ -= moveSpeed * g.camera.DirZ
		g.camera.PosX -= moveSpeed * g.camera.DirX
		g.camera.PosY -= moveSpeed * g.camera.DirY
	}
	if ebiten.IsKeyPressed(ebiten.KeyA) || ebiten.IsKeyPressed(ebiten.KeyLeft) {
		// Move left relative to direction (cross product with up vector)
		rightX := g.camera.DirY
		rightY := -g.camera.DirX
		g.camera.PosX -= moveSpeed * rightX
		g.camera.PosY -= moveSpeed * rightY
	}
	if ebiten.IsKeyPressed(ebiten.KeyD) || ebiten.IsKeyPressed(ebiten.KeyRight) {
		// Move right relative to direction (cross product with up vector)
		rightX := g.camera.DirY
		rightY := -g.camera.DirX
		g.camera.PosX += moveSpeed * rightX
		g.camera.PosY += moveSpeed * rightY
	}
	if ebiten.IsKeyPressed(ebiten.KeyQ) {
		g.camera.PosY -= moveSpeed // Move down
	}
	if ebiten.IsKeyPressed(ebiten.KeyE) {
		g.camera.PosY += moveSpeed // Move up
	}

	// Keyboard controls for rotation
	if ebiten.IsKeyPressed(ebiten.KeyJ) {
		// Rotate left around Y axis
		cosR := float32(math.Cos(float64(rotateSpeed)))
		sinR := float32(math.Sin(float64(rotateSpeed)))
		newDirX := g.camera.DirX*cosR - g.camera.DirZ*sinR
		g.camera.DirZ = g.camera.DirX*sinR + g.camera.DirZ*cosR
		g.camera.DirX = newDirX
	}
	if ebiten.IsKeyPressed(ebiten.KeyL) {
		// Rotate right around Y axis
		cosR := float32(math.Cos(float64(-rotateSpeed)))
		sinR := float32(math.Sin(float64(-rotateSpeed)))
		newDirX := g.camera.DirX*cosR - g.camera.DirZ*sinR
		g.camera.DirZ = g.camera.DirX*sinR + g.camera.DirZ*cosR
		g.camera.DirX = newDirX
	}
	if ebiten.IsKeyPressed(ebiten.KeyI) {
		// Rotate up
		cosR := float32(math.Cos(float64(rotateSpeed)))
		sinR := float32(math.Sin(float64(rotateSpeed)))
		newDirY := g.camera.DirY*cosR - g.camera.DirZ*sinR
		g.camera.DirZ = g.camera.DirY*sinR + g.camera.DirZ*cosR
		g.camera.DirY = newDirY
	}
	if ebiten.IsKeyPressed(ebiten.KeyK) {
		// Rotate down
		cosR := float32(math.Cos(float64(-rotateSpeed)))
		sinR := float32(math.Sin(float64(-rotateSpeed)))
		newDirY := g.camera.DirY*cosR - g.camera.DirZ*sinR
		g.camera.DirZ = g.camera.DirY*sinR + g.camera.DirZ*cosR
		g.camera.DirY = newDirY
	}

	// Mouse controls for rotation
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		g.camera.MouseDragging = true
		g.camera.LastMouseX, g.camera.LastMouseY = ebiten.CursorPosition()
	}
	if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		g.camera.MouseDragging = false
	}

	if g.camera.MouseDragging {
		currentX, currentY := ebiten.CursorPosition()
		dx := currentX - g.camera.LastMouseX
		dy := currentY - g.camera.LastMouseY

		// Rotate horizontally (around Y axis)
		if dx != 0 {
			rotateAngle := float32(dx) * mouseRotateSpeed
			cosR := float32(math.Cos(float64(-rotateAngle)))
			sinR := float32(math.Sin(float64(-rotateAngle)))
			newDirX := g.camera.DirX*cosR - g.camera.DirZ*sinR
			g.camera.DirZ = g.camera.DirX*sinR + g.camera.DirZ*cosR
			g.camera.DirX = newDirX
		}

		// Rotate vertically (pitch)
		if dy != 0 {
			rotateAngle := float32(dy) * mouseRotateSpeed
			cosR := float32(math.Cos(float64(-rotateAngle)))
			sinR := float32(math.Sin(float64(-rotateAngle)))
			newDirY := g.camera.DirY*cosR - g.camera.DirZ*sinR
			g.camera.DirZ = g.camera.DirY*sinR + g.camera.DirZ*cosR
			g.camera.DirY = newDirY
		}

		g.camera.LastMouseX = currentX
		g.camera.LastMouseY = currentY
	}

	// Normalize the direction vector
	length := float32(math.Sqrt(float64(g.camera.DirX*g.camera.DirX + g.camera.DirY*g.camera.DirY + g.camera.DirZ*g.camera.DirZ)))
	if length > 0 {
		g.camera.DirX /= length
		g.camera.DirY /= length
		g.camera.DirZ /= length
	}

	// Write camera data after any modifications
	g.writeCameraData()
}

func (g *Game) Update() error {
	// Handle camera movement
	g.handleCameraMovement()

	// Only update at 60 FPS to match the C program
	now := time.Now()
	if now.Sub(g.lastUpdate) >= frameDelay {
		err := g.UpdatePixels()
		if err != nil {
			fmt.Printf("Error updating pixels: %v\n", err)
		}
		g.lastUpdate = now
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	if g.img != nil {
		screen.DrawImage(g.img, nil)
	}
	PlotFPS(screen) // Add this line to draw the FPS graph
	ebitenutil.DebugPrint(screen, fmt.Sprintf("FPS: %.1f\nCamera: (%.1f, %.1f, %.1f) Dir: (%.1f, %.1f, %.1f)\nControls: WASD/Arrows - Move, QE - Up/Down, IJKL - Rotate, Mouse Drag - Look",
		ebiten.CurrentFPS(), g.camera.PosX, g.camera.PosY, g.camera.PosZ, g.camera.DirX, g.camera.DirY, g.camera.DirZ))
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenWidth, screenHeight
}

func main() {
	game := &Game{
		lastUpdate: time.Now(),
		camera: Camera{
			PosX: 50.0, PosY: 50.0, PosZ: -50.0, // Start a bit back from the particles
			DirX: 0.0, DirY: 0.0, DirZ: 1.0, // Looking forward
			FOV: 1.0,
		},
	}

	// Write initial camera data
	game.writeCameraData()

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Particle Viewer")
	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}
