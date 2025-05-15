package main

import (
	"bytes"
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
	moveSpeed        = 0.8
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

const (
	// Control mode: do not change these
	move   = uint8(iota)
	cursor = uint8(iota)
)

const (
	// New render modes: 0 = distance, 1 = velocity, 2 = normalized opacity
	renderDistance = uint8(iota)
	renderVelocity
	renderOpacity
)

type Game struct {
	pixels     [screenWidth * screenHeight * 4]uint8
	img        *ebiten.Image
	lastUpdate time.Time
	camera     Camera
	cursor     Cursor
	mode       uint8
	renderMode uint8 // will be one of renderDistance, renderVelocity, renderOpacity
}

type TimePartition struct {
	CollisionTime       int32
	ApplyPressureTime   int32
	UpdateParticlesTime int32
	MoveToBoxTime       int32
	UpdateGridTime      int32
	RenderTime          int32
}

func PlotTimePartition(screen *ebiten.Image) {
	// Plot parameters
	const (
		barHeight   = 20
		barSpacing  = 5
		barMaxWidth = 80
		startX      = 680
		startY      = screenHeight - 600
		labelWidth  = 80
	)

	// Read time partition data
	data, err := os.ReadFile("time_partition.bin")
	if err != nil {
		return
	}

	var tp TimePartition
	reader := bytes.NewReader(data)
	if err := binary.Read(reader, binary.LittleEndian, &tp); err != nil {
		return
	}

	// Define time segments with their colors
	segments := []struct {
		name  string
		time  int32
		color color.RGBA
	}{
		{"Collision", tp.CollisionTime, color.RGBA{255, 0, 0, 255}},
		{"Pressure", tp.ApplyPressureTime, color.RGBA{0, 255, 0, 255}},
		{"Update", tp.UpdateParticlesTime, color.RGBA{0, 0, 255, 255}},
		{"Box Bounds", tp.MoveToBoxTime, color.RGBA{255, 255, 0, 255}},
		{"Grid Update", tp.UpdateGridTime, color.RGBA{255, 0, 255, 255}},
		{"Render", tp.RenderTime, color.RGBA{0, 255, 255, 255}},
	}

	// Find maximum time for scaling
	maxTime := int32(1)
	for _, s := range segments {
		if s.time > maxTime {
			maxTime = s.time
		}
	}

	// Draw title
	ebitenutil.DebugPrintAt(screen, "Time Breakdown (ms):", startX, startY-30)

	// Draw bars
	for i, seg := range segments {
		y := startY + i*(barHeight+barSpacing)

		// Draw label
		ebitenutil.DebugPrintAt(screen, seg.name+":", startX-labelWidth, y+5)

		// Draw bar
		width := int(float64(seg.time) / float64(maxTime) * float64(barMaxWidth))
		if width < 2 {
			width = 2
		}
		ebitenutil.DrawRect(screen, float64(startX), float64(y), float64(width), float64(barHeight), seg.color)

		// Draw time value
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("%.1f", float32(seg.time)), int(startX+float64(width)+5), y+5)
	}
}

func PlotFPS(screen *ebiten.Image) {
	// Plot parameters
	const (
		samples     = 30                // Number of samples to plot
		graphX      = 35                // Left margin
		graphY      = screenHeight - 20 // Bottom margin
		graphWidth  = 180               // Width of graph
		graphHeight = 100               // Height of graph
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
	// Open file for binary reading
	data, err := os.ReadFile("output.bin")
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// 800x600 pixels, 3 bytes per pixel (distance, velocity, opacity)
	// if len(data) < screenWidth*screenHeight*3 {
	// 	return fmt.Errorf("file size is too small: %d bytes", len(data))
	// }
	// Each pixel in the file has 3 bytes: distance, velocity, and opacity
	for y := 0; y < screenHeight-1; y++ {
		for x := 0; x < screenWidth-1; x++ {
			srcPos := (y*screenWidth + x) * 3 // Changed from 4 to 3 bytes per pixel
			dstPos := (y*screenWidth + x) * 4

			if srcPos+3 > len(data) {
				continue // Skip if not enough data
			}
			var value uint8
			switch g.renderMode {
			case renderDistance:
				value = data[srcPos]
			case renderVelocity:
				value = data[srcPos+1]
			case renderOpacity:
				value = data[srcPos+2]
			}

			// Build a grayscale pixel based on the chosen channel
			if dstPos+3 <= len(g.pixels) {
				g.pixels[dstPos] = value   // R
				g.pixels[dstPos+1] = value // G
				g.pixels[dstPos+2] = value // B
				g.pixels[dstPos+3] = 255   // A
			}
		}
	}
	if g.img == nil {
		g.img = ebiten.NewImage(screenWidth, screenHeight)
	}
	g.img.WritePixels(g.pixels[:])
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

type Cursor struct {
	X, Y, Z    float32
	force      float32
	active     bool
	rightDrag  bool // Track right mouse button dragging
	lastMouseX int  // Store last mouse position for drag calculations
	lastMouseY int
}

func (c *Cursor) ReadFileData() error {
	// Read cursor data from a binary file
	data, err := os.ReadFile("cursor.bin")
	if err != nil {
		// If file doesn't exist, create it with default values
		if os.IsNotExist(err) {
			return c.WriteFileData() // Initialize with current values
		}
		return fmt.Errorf("failed to read cursor file: %w", err)
	}

	// Unpack the data into the cursor struct
	reader := bytes.NewReader(data)
	if err := binary.Read(reader, binary.LittleEndian, &c.X); err != nil {
		return fmt.Errorf("failed to read cursor X: %w", err)
	}
	if err := binary.Read(reader, binary.LittleEndian, &c.Y); err != nil {
		return fmt.Errorf("failed to read cursor Y: %w", err)
	}
	if err := binary.Read(reader, binary.LittleEndian, &c.Z); err != nil {
		return fmt.Errorf("failed to read cursor Z: %w", err)
	}
	if err := binary.Read(reader, binary.LittleEndian, &c.active); err != nil {
		return fmt.Errorf("failed to read cursor active: %w", err)
	}
	if err := binary.Read(reader, binary.LittleEndian, &c.force); err != nil {
		return fmt.Errorf("failed to read cursor force: %w", err)
	}

	return nil
}

func (c *Cursor) WriteFileData() error {
	// Create a binary file to share cursor data with the C program
	file, err := os.Create("cursor.bin")
	if err != nil {
		return fmt.Errorf("failed to create cursor file: %w", err)
	}
	defer file.Close()
	// Write cursor data in binary format
	binary.Write(file, binary.LittleEndian, c.X)
	binary.Write(file, binary.LittleEndian, c.Y)
	binary.Write(file, binary.LittleEndian, c.Z)
	binary.Write(file, binary.LittleEndian, c.active)
	binary.Write(file, binary.LittleEndian, c.force)
	return nil
}

func (c *Cursor) Update(g *Game) {
	// Force adjustment with scroll wheel
	_, scrollY := ebiten.Wheel()
	if scrollY != 0 {
		// Make force adjustment proportional to current force value for better control
		adjustment := math.Max(float64(c.force*0.1), 1.25)

		if scrollY > 0 {
			c.force += float32(adjustment)
		} else {
			c.force -= float32(adjustment)
		}
	}

	// Handle standard keyboard movement for cursor
	if ebiten.IsKeyPressed(ebiten.KeyW) || ebiten.IsKeyPressed(ebiten.KeyUp) {
		c.Z += moveSpeed * g.camera.DirZ
		c.X += moveSpeed * g.camera.DirX
		c.Y += moveSpeed * g.camera.DirY
	}
	if ebiten.IsKeyPressed(ebiten.KeyS) || ebiten.IsKeyPressed(ebiten.KeyDown) {
		c.Z -= moveSpeed * g.camera.DirZ
		c.X -= moveSpeed * g.camera.DirX
		c.Y -= moveSpeed * g.camera.DirY
	}
	if ebiten.IsKeyPressed(ebiten.KeyA) || ebiten.IsKeyPressed(ebiten.KeyLeft) {
		// Move left relative to direction (cross product with up vector)
		rightX := g.camera.DirY
		rightY := -g.camera.DirX
		c.X -= moveSpeed * rightX
		c.Y -= moveSpeed * rightY
	}
	if ebiten.IsKeyPressed(ebiten.KeyD) || ebiten.IsKeyPressed(ebiten.KeyRight) {
		// Move right relative to direction (cross product with up vector)
		rightX := g.camera.DirY
		rightY := -g.camera.DirX
		c.X += moveSpeed * rightX
		c.Y += moveSpeed * rightY
	}
	if ebiten.IsKeyPressed(ebiten.KeyQ) {
		c.Y -= moveSpeed // Move down
	}
	if ebiten.IsKeyPressed(ebiten.KeyE) {
		c.Y += moveSpeed // Move up
	}

	// Handle left click for activation
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		c.active = true
	} else if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonLeft) {
		c.active = false
	}

	// Handle right click for 3D cursor movement
	if g.mode == cursor {
		currentX, currentY := ebiten.CursorPosition()

		// Track right button state for dragging
		if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonRight) {
			c.rightDrag = true
			c.lastMouseX, c.lastMouseY = currentX, currentY
		} else if inpututil.IsMouseButtonJustReleased(ebiten.MouseButtonRight) {
			c.rightDrag = false
		}

		// If right button is being dragged, move cursor in 3D space
		if c.rightDrag {
			dx := currentX - c.lastMouseX
			dy := currentY - c.lastMouseY

			// Calculate camera's right vector (cross product of direction and up)
			upVector := [3]float32{0, 1, 0}
			rightX := g.camera.DirY*upVector[2] - g.camera.DirZ*upVector[1]
			rightY := g.camera.DirZ*upVector[0] - g.camera.DirX*upVector[2]
			rightZ := g.camera.DirX*upVector[1] - g.camera.DirY*upVector[0]

			// Normalize right vector
			rightLength := float32(math.Sqrt(float64(rightX*rightX + rightY*rightY + rightZ*rightZ)))
			if rightLength > 0 {
				rightX /= rightLength
				rightY /= rightLength
				rightZ /= rightLength
			}

			// Calculate camera's up vector (cross product of right and direction)
			upX := rightY*g.camera.DirZ - rightZ*g.camera.DirY
			upY := rightZ*g.camera.DirX - rightX*g.camera.DirZ
			upZ := rightX*g.camera.DirY - rightY*g.camera.DirX

			// Normalize up vector
			upLength := float32(math.Sqrt(float64(upX*upX + upY*upY + upZ*upZ)))
			if upLength > 0 {
				upX /= upLength
				upY /= upLength
				upZ /= upLength
			}

			// Adjust movement sensitivity based on distance from camera
			// This makes movement more intuitive at different distances
			dx_factor := float32(dx) * 0.1
			dy_factor := float32(dy) * 0.1

			// Apply horizontal movement (along right vector)
			c.X += rightX * dx_factor
			c.Y += rightY * dx_factor
			c.Z += rightZ * dx_factor

			// Apply vertical movement (along up vector)
			c.X -= upX * dy_factor
			c.Y -= upY * dy_factor
			c.Z -= upZ * dy_factor

			// Store current position for next frame
			c.lastMouseX, c.lastMouseY = currentX, currentY
		}
	}

	// Write cursor data after any modifications
	err := c.WriteFileData()
	if err != nil {
		fmt.Printf("Error writing cursor data: %v\n", err)
		return
	}
}

func (g *Game) handleCameraMovement() {
	// Keyboard controls for position
	if ebiten.IsKeyPressed(ebiten.KeyW) || ebiten.IsKeyPressed(ebiten.KeyUp) {
		g.camera.PosZ += moveSpeed * g.camera.DirZ
		g.camera.PosX += moveSpeed * g.camera.DirX
		g.camera.PosY += moveSpeed * g.camera.DirY
		// g.camera.PosZ += moveSpeed
	}
	if ebiten.IsKeyPressed(ebiten.KeyS) || ebiten.IsKeyPressed(ebiten.KeyDown) {
		g.camera.PosZ -= moveSpeed * g.camera.DirZ
		g.camera.PosX -= moveSpeed * g.camera.DirX
		g.camera.PosY -= moveSpeed * g.camera.DirY
		// g.camera.PosZ -= moveSpeed
	}
	if ebiten.IsKeyPressed(ebiten.KeyA) || ebiten.IsKeyPressed(ebiten.KeyLeft) {
		// Move left relative to direction (cross product with up vector)
		rightX := g.camera.DirY
		rightY := -g.camera.DirX
		g.camera.PosX -= moveSpeed * rightX
		g.camera.PosY -= moveSpeed * rightY
		// g.camera.PosX += moveSpeed
	}
	if ebiten.IsKeyPressed(ebiten.KeyD) || ebiten.IsKeyPressed(ebiten.KeyRight) {
		// Move right relative to direction (cross product with up vector)
		rightX := g.camera.DirY
		rightY := -g.camera.DirX
		g.camera.PosX += moveSpeed * rightX
		g.camera.PosY += moveSpeed * rightY
		// g.camera.PosX -= moveSpeed
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
	// Toggle render mode between distance, velocity and opacity
	if inpututil.IsKeyJustPressed(ebiten.KeyR) {
		g.renderMode = (g.renderMode + 1) % 3
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyTab) {
		g.mode = (g.mode + 1) % 2
	}


	switch g.mode {
	case move:
		g.handleCameraMovement()
	case cursor:
		g.cursor.Update(g)
	}

	// Only update at 60 FPS to match the C program
	now := time.Now()
	if now.Sub(g.lastUpdate) >= frameDelay {
		if err := g.UpdatePixels(); err != nil {
			fmt.Printf("Error updating pixels: %v\n", err)
		}
		g.lastUpdate = now
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Always draw g.img (which now represents the chosen channel)
	if g.img != nil {
		screen.DrawImage(g.img, nil)
	}
	PlotFPS(screen)
	PlotTimePartition(screen)
	ebitenutil.DebugPrint(screen, fmt.Sprintf("FPS: %.1f\nCamera: (%.1f, %.1f, %.1f) Dir: (%.1f, %.1f, %.1f)\nControls: WASD/Arrows - Move, QE - Up/Down, IJKL - Rotate, Mouse Drag - Look\nRender Mode: %s (Press R to change)",
		ebiten.CurrentFPS(), g.camera.PosX, g.camera.PosY, g.camera.PosZ,
		g.camera.DirX, g.camera.DirY, g.camera.DirZ,
		func() string {
			switch g.renderMode {
			case renderDistance:
				return "Distance"
			case renderVelocity:
				return "Velocity"
			case renderOpacity:
				return "Normalized Opacity"
			}
			return ""
		}()))
	if g.mode == move {
		ebitenutil.DebugPrintAt(screen, "Mode: Move", 0, 70)
	} else {
		ebitenutil.DebugPrintAt(screen, "Mode: Cursor", 0, 70)
	}
	// print force value
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Force: %.1f", g.cursor.force), 0, 90)
	// print render mode and how to change it
	if g.renderMode == renderDistance {
		ebitenutil.DebugPrintAt(screen, "Render Mode: Distance (Press R to cycle)", 0, 110)
	} else if g.renderMode == renderVelocity {
		ebitenutil.DebugPrintAt(screen, "Render Mode: Velocity (Press R to cycle)", 0, 110)
	} else {
		ebitenutil.DebugPrintAt(screen, "Render Mode: Opacity (Press R to cycle)", 0, 110)
	}
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
		mode: move, // Start in movement mode
		cursor: Cursor{
			X: 0.0, Y: 0.0, Z: 0.0,
			active:    false,
			force:     100.0,
			rightDrag: false,
		},
		renderMode: renderDistance, // Use renderDistance instead of undefined normalRender
	}

	// write cursor data to file
	err := game.cursor.WriteFileData()
	if err != nil {
		fmt.Printf("Error writing cursor data: %v\n", err)
	}

	// Write initial camera data
	game.writeCameraData()

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Particle Viewer")
	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}
