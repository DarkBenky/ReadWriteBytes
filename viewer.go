package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image/color"
	"io/ioutil"
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
	frameDelay   = 35 * time.Millisecond // Match the C program's 60 FPS

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
	renderNormal
	renderFluid
)

type Game struct {
	cameraImageData                    *CameraImageData // Struct to hold camera image data
	pixels                             [screenWidth * screenHeight * 4]uint8
	pixelsVelocity                     [screenWidth * screenHeight * 4]uint8
	pixelsOpacity                      [screenWidth * screenHeight * 4]uint8
	pixelsDistance                     [screenWidth * screenHeight * 4]uint8
	pixelsOpacityFromCameraPerspective [screenWidth * screenHeight * 4]uint8
	shaders                            []Shader
	normalShader                       *ebiten.Shader
	gaussianBlurShader                 *ebiten.Shader
	waterShader                        *ebiten.Shader
	mixShader                          *ebiten.Shader
	img                                *ebiten.Image
	imgVelocity                        *ebiten.Image
	imgOpacity                         *ebiten.Image
	imgDistance                        *ebiten.Image
	selectedOption                     string
	optionChangeRate                   float64
	lastUpdate                         time.Time
	camera                             Camera
	cursor                             Cursor
	mode                               uint8
	renderMode                         uint8 // will be one of renderDistance, renderVelocity, renderOpacity
	selectedShader                     uint8
	pause                              bool
	CameraOrLightPosition              bool
}

//	struct TimePartition {
//	    int collisionTime;
//	    int applyPressureTime;
//	    int updateParticlesTime;
//	    int moveToBoxTime;
//	    int updateGridTime;
//	    int renderTime;
//	    int clearScreenTime;
//	    int projectParticlesTime;
//	    int drawCursorTime;
//	    int drawBoundingBoxTime;
//	    int saveScreenTime;
//		int sortTime;
//		int projectionTime;
//		int renderDistanceVelocityTime;
//		int renderOpacityTime;
//	};
type TimePartition struct {
	CollisionTime              float32
	ApplyPressureTime          float32
	UpdateParticlesTime        float32
	MoveToBoxTime              float32
	UpdateGridTime             float32
	RenderTime                 float32
	ClearScreenTime            float32
	ProjectParticlesTime       float32
	DrawCursorTime             float32
	DrawBoundingBoxTime        float32
	SaveScreenTime             float32
	SortTime                   float32
	ProjectionTime             float32
	RenderDistanceVelocityTime float32
	RenderOpacityTime          float32
	ReadDataTime               float32
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
		time  float32
		color color.RGBA
	}{
		{"Read Data", tp.ReadDataTime, color.RGBA{128, 255, 255, 255}},
		{"Collision", tp.CollisionTime, color.RGBA{255, 0, 0, 255}},
		{"Pressure", tp.ApplyPressureTime, color.RGBA{0, 255, 0, 255}},
		{"Update", tp.UpdateParticlesTime, color.RGBA{0, 0, 255, 255}},
		{"Box Bounds", tp.MoveToBoxTime, color.RGBA{255, 255, 0, 255}},
		{"Grid Update", tp.UpdateGridTime, color.RGBA{255, 0, 255, 255}},
		{"Render", tp.RenderTime, color.RGBA{0, 255, 255, 255}},
		{"Clear Screen", tp.ClearScreenTime, color.RGBA{255, 128, 0, 255}},
		{"Project Particles", tp.ProjectParticlesTime, color.RGBA{128, 255, 0, 255}},
		{"Draw Cursor", tp.DrawCursorTime, color.RGBA{128, 0, 255, 255}},
		{"Draw Box", tp.DrawBoundingBoxTime, color.RGBA{0, 128, 255, 255}},
		{"Save Screen", tp.SaveScreenTime, color.RGBA{128, 128, 128, 255}},
		{"Sort", tp.SortTime, color.RGBA{255, 128, 128, 255}},
		{"Projection", tp.ProjectionTime, color.RGBA{128, 255, 128, 255}},
		{"Render Dist/Velo", tp.RenderDistanceVelocityTime, color.RGBA{128, 128, 255, 255}},
		{"Render Opacity", tp.RenderOpacityTime, color.RGBA{255, 128, 255, 255}},
	}

	// Find maximum time for scaling
	maxTime := float32(0)
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
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("%.5f", float32(seg.time)), int(startX+float64(width)+5), y+5)
	}
	// Print the total time and FPS of the C program
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Total Time: %.5f s", tp.CollisionTime+tp.ApplyPressureTime+tp.UpdateParticlesTime+tp.MoveToBoxTime+tp.UpdateGridTime+tp.RenderTime+tp.ClearScreenTime+tp.ProjectParticlesTime+tp.DrawCursorTime+tp.DrawBoundingBoxTime+tp.SaveScreenTime), startX-80, startY+len(segments)*(barHeight+barSpacing)+10)
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("FPS: %.3f", 1/(tp.CollisionTime+tp.ApplyPressureTime+tp.UpdateParticlesTime+tp.MoveToBoxTime+tp.UpdateGridTime+tp.RenderTime+tp.ClearScreenTime+tp.ProjectParticlesTime+tp.DrawCursorTime+tp.DrawBoundingBoxTime+tp.SaveScreenTime)), startX-80, startY+len(segments)*(barHeight+barSpacing)+30)
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

// func (g *Game) CalculateNormalShader() {
// 	const chucks = 8
// 	wg := sync.WaitGroup{}
// 	const chuckSize = screenWidth * screenHeight * 4 / chucks
// 	for i := 0; i < chucks; i++ {
// 		wg.Add(1)
// 		go func(start, end int) {
// 			for j := start; j < end; j += 16 {
// 				// get left pixel
// 				left := j - 4
// 				// get right pixel
// 				right := j + 4
// 				// get top pixel
// 				top := j - screenWidth*4
// 				// get bottom pixel
// 				bottom := j + screenWidth*4

// 				// calculate gradient
// 				gradientX := float32(0)
// 				gradientY := float32(0)
// 				if left >= 0 {
// 					gradientX += float32(g.pixels[left]) - float32(g.pixels[j])
// 				}
// 				if right < len(g.pixels) {
// 					gradientX += float32(g.pixels[right]) - float32(g.pixels[j])
// 				}
// 				if top >= 0 {
// 					gradientY += float32(g.pixels[top]) - float32(g.pixels[j])
// 				}
// 				if bottom < len(g.pixels) {
// 					gradientY += float32(g.pixels[bottom]) - float32(g.pixels[j])
// 				}
// 				// calculate normal
// 				normalX := gradientX
// 				normalY := gradientY
// 				normalZ := float32(1.0) // Assuming a constant Z value for simplicity
// 				// normalize normal
// 				length := float32(math.Sqrt(float64(normalX*normalX + normalY*normalY + normalZ*normalZ)))
// 				if length > 0 {
// 					normalX /= length
// 					normalY /= length
// 					normalZ /= length
// 				}
// 				// store normal in NormalBuffer
// 				NormalBuffer[j] = uint8((normalX + 1) * 127.5)
// 			}
// 		}(i*chuckSize, (i+1)*chuckSize)
// 	}
// 	fmt.Println("Normal shader calculation started")
// 	wg.Wait()
// 	fmt.Println("Normal shader calculation finished")
// 	g.img.WritePixels(NormalBuffer[:])
// }

type CameraImageData struct {
	pixelsDistance [screenWidth * screenHeight * 4]uint8
	pixelsVelocity [screenWidth * screenHeight * 4]uint8
	pixelsOpacity  [screenWidth * screenHeight * 4]uint8
	imgDistance    *ebiten.Image
	imgVelocity    *ebiten.Image
	imgOpacity     *ebiten.Image
}

// func (g *Game) UpdatePixelsGeneral(fileName string) error {
// 	// Open file for binary reading
// 	data, err := os.ReadFile(fileName)
// 	if err != nil {
// 		return fmt.Errorf("failed to read file: %w", err)
// 	}

// 	for y := 0; y < screenHeight-1; y++ {
// 		for x := 0; x < screenWidth-1; x++ {
// 			srcPos := (y*screenWidth + x) * 3 // Changed from 4 to 3 bytes per pixel
// 			dstPos := (y*screenWidth + x) * 4

// 			if srcPos+3 > len(data) {
// 				continue // Skip if not enough data
// 			}

// 			// Fill all 4 channels (RGBA) for each buffer
// 			for i := 0; i < 3; i++ {
// 				if dstPos+i < len(g.cameraImageData.pixelsDistance) {
// 					g.cameraImageData.pixelsDistance[dstPos+i] = data[srcPos]   // Distance
// 					g.cameraImageData.pixelsVelocity[dstPos+i] = data[srcPos+1] // Velocity
// 					g.cameraImageData.pixelsOpacity[dstPos+i] = data[srcPos+2]  // Opacity
// 				}
// 			}
// 			// Set alpha channel
// 			if dstPos+3 < len(g.cameraImageData.pixelsDistance) {
// 				g.cameraImageData.pixelsDistance[dstPos+3] = 255
// 				g.cameraImageData.pixelsVelocity[dstPos+3] = 255
// 				g.cameraImageData.pixelsOpacity[dstPos+3] = 255
// 			}
// 		}
// 	}

// 	if g.cameraImageData.imgDistance == nil {
// 		g.cameraImageData.imgDistance = ebiten.NewImage(screenWidth, screenHeight)
// 	}
// 	if g.cameraImageData.imgVelocity == nil {
// 		g.cameraImageData.imgVelocity = ebiten.NewImage(screenWidth, screenHeight)
// 	}
// 	if g.cameraImageData.imgOpacity == nil {
// 		g.cameraImageData.imgOpacity = ebiten.NewImage(screenWidth, screenHeight)
// 	}
// 	g.cameraImageData.imgDistance.WritePixels(g.cameraImageData.pixelsDistance[:])
// 	g.cameraImageData.imgVelocity.WritePixels(g.cameraImageData.pixelsVelocity[:])
// 	g.cameraImageData.imgOpacity.WritePixels(g.cameraImageData.pixelsOpacity[:])
// 	return nil
// }

func (g *Game) UpdatePixels() error {
	// Open file for binary reading
	data, err := os.ReadFile("output.bin")
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// Each pixel in the file has 3 bytes: distance, velocity, and opacity
	for y := 0; y < screenHeight-1; y++ {
		for x := 0; x < screenWidth-1; x++ {
			srcPos := (y*screenWidth + x) * 3 // 3 bytes per pixel
			dstPos := (y*screenWidth + x) * 4 // 4 bytes per pixel for RGBA

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
			case renderNormal:
				// Don't use the main data for normals, we'll read normal.bin separately
				value = 0
			case renderFluid:
				value = data[srcPos] // Use distance for grayscale
				for i := 0; i < 3; i++ {
					g.pixelsOpacity[dstPos+i] = data[srcPos+2]
					g.pixelsVelocity[dstPos+i] = data[srcPos+1]
					g.pixelsDistance[dstPos+i] = data[srcPos]
				}
			}

			// Build a grayscale pixel based on the chosen channel (except for normals)
			if g.renderMode != renderNormal && dstPos+3 <= len(g.pixels) {
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

	// Handle normal rendering separately
	if g.renderMode == renderNormal {
		normalData, err := ioutil.ReadFile("normal.bin")
		if err != nil {
			return fmt.Errorf("failed to read normal file: %w", err)
		}

		// Check if we have 3 bytes per pixel (RGB) or 4 bytes per pixel (RGBA)
		expectedSize4 := screenWidth * screenHeight * 4 // RGBA format

		if len(normalData) >= expectedSize4 {
			// File has RGBA format (4 bytes per pixel)
			g.img.WritePixels(normalData[:expectedSize4])
		}
	} else {
		// For other render modes, handle fluid rendering and write pixels normally
		if renderFluid == g.renderMode {
			if g.imgVelocity == nil {
				g.imgVelocity = ebiten.NewImage(screenWidth, screenHeight)
			}
			if g.imgOpacity == nil {
				g.imgOpacity = ebiten.NewImage(screenWidth, screenHeight)
			}
			if g.imgDistance == nil {
				g.imgDistance = ebiten.NewImage(screenWidth, screenHeight)
			}

			g.imgVelocity.WritePixels(g.pixelsVelocity[:])
			g.imgOpacity.WritePixels(g.pixelsOpacity[:])
			g.imgDistance.WritePixels(g.pixelsDistance[:])
		}

		g.img.WritePixels(g.pixels[:])
	}

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
		g.renderMode = (g.renderMode + 1) % 5
		fmt.Println("Render mode changed to", g.renderMode)
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyTab) {
		g.mode = (g.mode + 1) % 2
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyC) {
		fmt.Println("C key pressed, toggling camera/light position")
		g.CameraOrLightPosition = !g.CameraOrLightPosition // Toggle between camera and light position
	}

	// select shader with number keys
	// up
	if inpututil.IsKeyJustPressed(ebiten.KeyM) {
		g.selectedShader = g.selectedShader + 1
		fmt.Println("selectedShader", g.selectedShader, g.selectedShader)
		if g.selectedShader >= uint8(len(g.shaders)) {
			g.selectedShader = 0
		}
	} else if inpututil.IsKeyJustPressed(ebiten.KeyN) {
		g.selectedShader = g.selectedShader - 1
		fmt.Println("selectedShader", g.selectedShader, g.selectedShader)
		if g.selectedShader < 0 {
			g.selectedShader = uint8(len(g.shaders) - 1)
		}
	}

	// Toggle shader options with O key
	if inpututil.IsKeyJustPressed(ebiten.KeyO) {
		g.cycleShaderOption()
	}

	// Adjust selected option with scroll wheel when not adjusting cursor force
	if g.selectedOption != "" && g.mode != cursor {
		_, scrollY := ebiten.Wheel()
		if scrollY != 0 && int(g.selectedShader) < len(g.shaders) {
			shader := &g.shaders[g.selectedShader]
			if val, ok := shader.options[g.selectedOption]; ok {
				// Adjust by 7.5% of the current value, or a minimum amount
				var change float64
				switch v := val.(type) {
				case float64:
					change = v * 0.075 * float64(scrollY)
					if math.Abs(change) < 0.01 {
						change = 0.01 * math.Copysign(1, float64(scrollY))
					}
					shader.options[g.selectedOption] = v + change
				case float32:
					change = float64(v) * 0.075 * float64(scrollY)
					if math.Abs(change) < 0.01 {
						change = 0.01 * math.Copysign(1, float64(scrollY))
					}
					shader.options[g.selectedOption] = v + float32(change)
				case int:
					change = float64(v) * 0.075 * float64(scrollY)
					if math.Abs(change) < 1 {
						change = 1 * math.Copysign(1, float64(scrollY))
					}
					shader.options[g.selectedOption] = v + int(change)
				}
			}
		}
	}

	switch g.mode {
	case move:
		g.handleCameraMovement()
	case cursor:
		g.cursor.Update(g)
	}

	// check if p is pressed to pause
	if inpututil.IsKeyJustPressed(ebiten.KeyP) {
		g.pause = !g.pause
	}

	// Only update at 60 FPS to match the C program
	now := time.Now()
	if now.Sub(g.lastUpdate) >= frameDelay {
		if err := g.UpdatePixels(); err != nil {
			fmt.Printf("Error updating pixels: %v\n", err)
		}
		// fmt.Println("Pixels updated")
		// if err := g.UpdatePixelsGeneral("light.bin"); err != nil {
		// 	fmt.Printf("Error updating camera pixels: %v\n", err)
		// }
		// fmt.Println("Camera pixels updated")
		g.lastUpdate = now

		// write pause state to file
		// write the pause state to a file pause.bin
		file, err := os.Create("pause.bin")
		if err != nil {
			return fmt.Errorf("failed to create pause file: %w", err)
		}
		defer file.Close()
		if err := binary.Write(file, binary.LittleEndian, g.pause); err != nil {
			return fmt.Errorf("failed to write pause state: %w", err)
		}
		// fmt.Println("Pause state written to file:", g.pause)
	}
	return nil
}

func ApplyShader(image *ebiten.Image, shader *Shader) *ebiten.Image {
	if image == nil {
		return nil
	}

	newImage := ebiten.NewImageFromImage(image)
	opts := &ebiten.DrawRectShaderOptions{}
	opts.Images[0] = image
	opts.Uniforms = shader.options

	// Apply the shader
	newImage.DrawRectShader(
		newImage.Bounds().Dx(),
		newImage.Bounds().Dy(),
		shader.shader,
		opts,
	)

	return newImage
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Always draw g.img (which now represents the chosen channel)

	if g.CameraOrLightPosition {
		if g.img != nil {
			// apply shaders
			for _, shader := range g.shaders {
				g.img = ApplyShader(g.img, &shader)
			}

			if g.renderMode == renderNormal {
				// newImage := ebiten.NewImageFromImage(g.img)

				// opts := &ebiten.DrawRectShaderOptions{}
				// opts.Images[0] = g.img
				// // assign the camera direction to the shader options
				// opts.Uniforms = map[string]interface{}{
				// 	"cameraDirX": g.camera.DirX,
				// 	"cameraDirY": g.camera.DirY,
				// 	"cameraDirZ": g.camera.DirZ,
				// }
				// newImage.DrawRectShader(
				// 	newImage.Bounds().Dx(),
				// 	newImage.Bounds().Dy(),
				// 	g.normalShader,
				// 	opts,
				// )
				// g.img = newImage
			}
			if g.renderMode == renderFluid {
				newImage := ebiten.NewImageFromImage(g.img)

				opts := &ebiten.DrawRectShaderOptions{}
				opts.Images[0] = g.img
				opts.Images[1] = g.imgOpacity
				opts.Images[2] = g.imgDistance
				// assign the camera direction to the shader options
				opts.Uniforms = map[string]interface{}{
					"cameraDirX":       g.camera.DirX,
					"cameraDirY":       g.camera.DirY,
					"cameraDirZ":       g.camera.DirZ,
					"CameraPosX":       g.camera.PosX,
					"CameraPosY":       g.camera.PosY,
					"CameraPosZ":       g.camera.PosZ,
					"LightAbsorptionR": 1.5,
					"LightAbsorptionG": 2.2,
					"LightAbsorptionB": 10.8,
				}
				newImage.DrawRectShader(
					newImage.Bounds().Dx(),
					newImage.Bounds().Dy(),
					g.waterShader,
					opts,
				)
				g.img = newImage
			}
			screen.DrawImage(g.img, nil)
		} else {
			// screen.DrawImage(g.cameraImageData.imgOpacity, nil)
			switch g.renderMode {
			case renderDistance:
				tempImg := g.cameraImageData.imgDistance
				for _, shader := range g.shaders {
					tempImg = ApplyShader(tempImg, &shader)
				}
				screen.DrawImage(tempImg, nil)
				fmt.Println("Drawing distance image")
			case renderVelocity:
				tempImg := g.cameraImageData.imgVelocity
				for _, shader := range g.shaders {
					tempImg = ApplyShader(tempImg, &shader)
				}
				screen.DrawImage(tempImg, nil)
				fmt.Println("Drawing velocity image")
			case renderOpacity:
				tempImg := g.cameraImageData.imgOpacity
				for _, shader := range g.shaders {
					tempImg = ApplyShader(tempImg, &shader)
				}
				screen.DrawImage(tempImg, nil)
				fmt.Println("Drawing opacity image")
			case renderNormal:
				tempImg := g.cameraImageData.imgDistance
				for _, shader := range g.shaders {
					tempImg = ApplyShader(tempImg, &shader)
				}
				newImage := ebiten.NewImageFromImage(tempImg)

				opts := &ebiten.DrawRectShaderOptions{}
				opts.Images[0] = tempImg
				// assign the camera direction to the shader options
				opts.Uniforms = map[string]interface{}{
					"cameraDirX": g.camera.DirX,
					"cameraDirY": g.camera.DirY,
					"cameraDirZ": g.camera.DirZ,
				}
				newImage.DrawRectShader(
					newImage.Bounds().Dx(),
					newImage.Bounds().Dy(),
					g.normalShader,
					opts,
				)
				screen.DrawImage(newImage, nil)
				fmt.Println("Drawing normal image")
			}
		}

		// screen.DrawImage(g.cameraImageData.imgDistance, nil)
		// screen.DrawImage(g.cameraImageData.imgVelocity, nil)
		// screen.DrawImage(g.cameraImageData.imgOpacity, nil)
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
			case renderNormal:
				return "Normal"
			case renderFluid:
				return "Fluid"
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
	} else if g.renderMode == renderOpacity {
		ebitenutil.DebugPrintAt(screen, "Render Mode: Opacity (Press R to cycle)", 0, 110)
	} else if g.renderMode == renderNormal {
		ebitenutil.DebugPrintAt(screen, "Render Mode: Normal (Press R to cycle)", 0, 110)
	} else if g.renderMode == renderFluid {
		ebitenutil.DebugPrintAt(screen, "Render Mode: Fluid (Press R to cycle)", 0, 110)
	} else {
		ebitenutil.DebugPrintAt(screen, "Render Mode: Unknown", 0, 110)
	}
	// print shaders and how to change them
	ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Shader Menu (Press M/N to cycle, O to select option, scroll to adjust)"), 0, 130)
	for i, shader := range g.shaders {
		optionsStr := "{"
		for key, value := range shader.options {
			if i == int(g.selectedShader) && key == g.selectedOption {
				optionsStr += fmt.Sprintf("[%s: %v], ", key, value) // Highlight selected option
			} else {
				optionsStr += fmt.Sprintf("%s: %v, ", key, value)
			}
		}
		if len(optionsStr) > 1 {
			optionsStr = optionsStr[:len(optionsStr)-2] // Remove trailing comma and space
		}
		optionsStr += "}"
		if i == int(g.selectedShader) {
			ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Shader %d: %s shader options %s -> Selected", i+1, shader.name, optionsStr), 0, 150+i*20)
		} else {
			ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Shader %d: %s shader options %s", i+1, shader.name, optionsStr), 0, 150+i*20)
		}
	}
	// print if the camera or light position is being rendered
	if g.CameraOrLightPosition {
		ebitenutil.DebugPrintAt(screen, "Rendering Camera Position (Switch with C)", 0, 200)
	} else {
		ebitenutil.DebugPrintAt(screen, "Rendering Light Position (Switch with C)", 0, 200)
	}

}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenWidth, screenHeight
}

type Shader struct {
	shader  *ebiten.Shader
	options map[string]interface{}
	name    string
}

func loadShader(filePath string) ([]byte, error) {
	// Load shader from file
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read shader file: %w", err)
	}
	// Check if the shader is valid
	if len(data) == 0 {
		return nil, fmt.Errorf("shader file is empty")
	}
	return data, nil
}

func (g *Game) cycleShaderOption() {
	if len(g.shaders) == 0 || g.selectedShader >= uint8(len(g.shaders)) {
		return
	}

	shader := &g.shaders[g.selectedShader]
	if len(shader.options) == 0 {
		g.selectedOption = ""
		return
	}

	// Get all option keys
	keys := make([]string, 0, len(shader.options))
	for k := range shader.options {
		keys = append(keys, k)
	}

	// If no option is selected, select the first one
	if g.selectedOption == "" {
		g.selectedOption = keys[0]
		return
	}

	// Find current option and move to next
	for i, key := range keys {
		if key == g.selectedOption {
			nextIndex := (i + 1) % len(keys)
			g.selectedOption = keys[nextIndex]
			return
		}
	}

	// If we get here, the selected option doesn't exist anymore
	g.selectedOption = keys[0]
}

func main() {
	// var SigmaSpatial float  // Controls the influence of distance
	// var SigmaRange float    // Controls the influence of color difference
	src, err := loadShader("shaders/blur.kage")
	if err != nil {
		fmt.Printf("Error loading shader: %v\n", err)
		panic(err)
	}
	// blurShader, err := ebiten.NewShader(src)
	// if err != nil {
	// 	fmt.Printf("Error creating shader: %v\n", err)
	// 	panic(err)
	// }

	src, err = loadShader("shaders/calculateNormal.kage")
	if err != nil {
		fmt.Printf("Error loading shader: %v\n", err)
		panic(err)
	}
	calculateNormalShader, err := ebiten.NewShader(src)
	if err != nil {
		fmt.Printf("Error creating shader: %v\n", err)
		panic(err)
	}

	src, err = loadShader("shaders/gaussianBlur.kage")
	if err != nil {
		fmt.Printf("Error loading shader: %v\n", err)
		panic(err)
	}
	gaussianBlurShader, err := ebiten.NewShader(src)
	if err != nil {
		fmt.Printf("Error creating shader: %v\n", err)
		panic(err)
	}

	waterShaderSrc, err := loadShader("shaders/water.kage")
	if err != nil {
		fmt.Printf("Error loading water shader: %v\n", err)
		panic(err)
	}
	waterShader, err := ebiten.NewShader(waterShaderSrc)
	if err != nil {
		fmt.Printf("Error creating water shader: %v\n", err)
		panic(err)
	}

	mixShaderSrc, err := loadShader("shaders/encodeRGBplusA.kage")
	if err != nil {
		fmt.Printf("Error loading mix shader: %v\n", err)
		panic(err)
	}
	mixShader, err := ebiten.NewShader(mixShaderSrc)
	if err != nil {
		fmt.Printf("Error creating mix shader: %v\n", err)
		panic(err)
	}

	// src, err = loadShader("shaders/example.kage")
	// if err != nil {
	// 	fmt.Printf("Error loading shader: %v\n", err)
	// 	panic(err)
	// }
	// example, err := ebiten.NewShader(src)
	// if err != nil {
	// 	fmt.Printf("Error creating shader: %v\n", err)
	// 	panic(err)
	// }

	shaders := []Shader{
		// {shader: blurShader, options: map[string]interface{}{"SigmaSpatial": 2.0, "SigmaRange": 1.5}, name: "Blur"},
		// {shader: blurShader, options: map[string]interface{}{"SigmaSpatial": 0.75, "SigmaRange": 0.5}, name: "Blur"},
		// {shader: blurShader, options: map[string]interface{}{"SigmaSpatial": 0.85, "SigmaRange": 0.5}, name: "Blur"},
		// {shader: blurShader, options: map[string]interface{}{"SigmaSpatial": 1.05, "SigmaRange": 0.5}, name: "Blur"},
	}

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
			force:     10.0,
			rightDrag: false,
		},
		renderMode:            renderDistance,
		shaders:               shaders,
		selectedShader:        0,
		selectedOption:        "",
		optionChangeRate:      0.075, // 7.5% change per scroll event
		normalShader:          calculateNormalShader,
		gaussianBlurShader:    gaussianBlurShader,
		waterShader:           waterShader,
		cameraImageData:       &CameraImageData{},
		CameraOrLightPosition: true,
		mixShader:             mixShader,
	}

	// write cursor data to file
	err = game.cursor.WriteFileData()
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
