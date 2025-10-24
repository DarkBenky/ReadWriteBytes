from noise import pnoise2
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import LinearSegmentedColormap

MAP_SIZE = 205 # 15 km
ITERATIONS = 3
HEIGHT_MULTIPLIER = 0.45

def generate_enhanced_height_map(size=MAP_SIZE, seed=None):
    if seed is not None:
        random.seed(seed)
    
    height_map = np.zeros((size, size))
    
    print("Generating base terrain...")
    frequency = 0.008
    amplitude = 0.6
    persistence = 0.4
    lacunarity = 2.0
    
    for i in range(size):
        for j in range(size):
            noise_value = 0
            freq = frequency
            amp = amplitude
            
            for octave in range(4):
                noise_value += pnoise2(i * freq, j * freq, octaves=1) * amp
                freq *= lacunarity
                amp *= persistence
            
            height_map[i][j] = noise_value
    
    print("Adding mountain ridges...")
    ridge_frequency = 0.003
    for i in range(size):
        for j in range(size):
            ridge_noise = abs(pnoise2(i * ridge_frequency, j * ridge_frequency, octaves=1))
            ridge_height = 1.0 - ridge_noise
            ridge_height = ridge_height ** 2
            height_map[i][j] += ridge_height * 0.3
    
    print("Carving river valleys...")
    valley_frequency = 0.006
    for i in range(size):
        for j in range(size):
            valley_noise = abs(pnoise2(i * valley_frequency + 1000, j * valley_frequency + 1000, octaves=1))
            valley_depth = (1.0 - valley_noise) ** 3
            height_map[i][j] -= valley_depth * 0.15
    
    print("Adding surface details...")
    detail_frequency = 0.03
    for i in range(size):
        for j in range(size):
            detail = pnoise2(i * detail_frequency, j * detail_frequency, octaves=2)
            height_map[i][j] += detail * 0.05
    
    print("Applying erosion effects...")
    height_map = apply_thermal_erosion(height_map, iterations=ITERATIONS, talus_angle=0.12)
    
    height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
    height_map = height_map * 1.2 - 0.3
    
    return height_map

def apply_thermal_erosion(height_map, iterations=1, talus_angle=0.08):
    height_map = height_map.copy()
    
    for _ in range(iterations):
        for i in range(1, height_map.shape[0] - 1):
            for j in range(1, height_map.shape[1] - 1):
                neighbors = [
                    height_map[i-1, j], height_map[i+1, j],
                    height_map[i, j-1], height_map[i, j+1],
                    height_map[i-1, j-1], height_map[i-1, j+1],
                    height_map[i+1, j-1], height_map[i+1, j+1]
                ]
                
                max_diff = 0
                for neighbor in neighbors:
                    diff = height_map[i, j] - neighbor
                    if diff > max_diff:
                        max_diff = diff
                
                if max_diff > talus_angle:
                    height_map[i, j] -= (max_diff - talus_angle) * 0.05
    
    return height_map

def generate_height_map(size=MAP_SIZE):
    return generate_enhanced_height_map(size)

def create_custom_colormap():
    colors = [
        '#001133',  # Deep water (dark blue)
        '#003366',  # Shallow water (blue)
        '#4A90E2',  # Water (light blue)
        '#87CEEB',  # Shore water (sky blue)
        '#F4E4BC',  # Beach/sand (beige)
        '#228B22',  # Lowlands (green)
        '#32CD32',  # Plains (light green)
        '#9ACD32',  # Hills (yellow-green)
        '#8B4513',  # Mountains (brown)
        '#A0522D',  # High mountains (dark brown)
        '#696969',  # Peaks (gray)
        '#FFFFFF'   # Snow caps (white)
    ]
    
    return LinearSegmentedColormap.from_list('realistic_terrain', colors, N=256)

def display_height_map(height_map, title="Height Map", save_path=None):
    height_array = np.array(height_map)
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    custom_cmap = create_custom_colormap()
    im1 = ax1.imshow(height_array, cmap=custom_cmap, interpolation='bilinear')
    ax1.set_title(f'{title} - Terrain View', fontsize=16, color='white')
    ax1.set_xlabel('X coordinate (km)', color='white')
    ax1.set_ylabel('Y coordinate (km)', color='white')
    ax1.tick_params(colors='white')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Elevation', color='white', fontsize=12)
    cbar1.ax.tick_params(colors='white')
    
    contour_levels = 20
    im2 = ax2.contourf(height_array, levels=contour_levels, cmap='viridis', alpha=0.8)
    contours = ax2.contour(height_array, levels=contour_levels, colors='white', alpha=0.4, linewidths=0.5)
    ax2.clabel(contours, inline=True, fontsize=8, colors='white')
    
    ax2.set_title(f'{title} - Topographic View', fontsize=16, color='white')
    ax2.set_xlabel('X coordinate (km)', color='white')
    ax2.set_ylabel('Y coordinate (km)', color='white')
    ax2.tick_params(colors='white')
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Elevation', color='white', fontsize=12)
    cbar2.ax.tick_params(colors='white')
    
    stats_text = f"""Map Statistics:
Min Elevation: {np.min(height_array):.3f}
Max Elevation: {np.max(height_array):.3f}
Mean Elevation: {np.mean(height_array):.3f}
Std Deviation: {np.std(height_array):.3f}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, color='white', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Enhanced map saved to {save_path}")
    
    plt.show()

def generate_3d_view(height_map, title="3D Terrain View", save_path=None):
    height_array = np.array(height_map)
    
    if height_array.shape[0] > 200:
        step = height_array.shape[0] // 200
        height_array = height_array[::step, ::step]
    
    x = np.arange(height_array.shape[1])
    y = np.arange(height_array.shape[0])
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surface = ax.plot_surface(X, Y, height_array, 
                            cmap=create_custom_colormap(),
                            alpha=0.9, 
                            linewidth=0, 
                            antialiased=True,
                            shade=True)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Elevation')
    
    fig.colorbar(surface, shrink=0.6, aspect=30)
    
    ax.view_init(elev=45, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D view saved to {save_path}")
    
    plt.show()


class Triangle:
    def __init__(self, v1x, v1y, v1z,
                 v2x, v2y, v2z,
                 v3x, v3y, v3z,
                 normalx, normaly, normalz,
                 colorR, colorG, colorB):
        self.v1x = v1x
        self.v1y = v1y
        self.v1z = v1z
        self.v2x = v2x
        self.v2y = v2y
        self.v2z = v2z
        self.v3x = v3x
        self.v3y = v3y
        self.v3z = v3z
        self.normalx = normalx
        self.normaly = normaly
        self.normalz = normalz
        self.roughness = 0.95
        self.metallic = 0.05
        self.emission = 0.05
        self.colorR = colorR
        self.colorG = colorG
        self.colorB = colorB

def compute_normal(v1, v2, v3):
    u = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
    v = (v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2])
    
    nx = u[1] * v[2] - u[2] * v[1]
    ny = u[2] * v[0] - u[0] * v[2]
    nz = u[0] * v[1] - u[1] * v[0]
    
    length = (nx**2 + ny**2 + nz**2) ** 0.5
    if length == 0:
        return (0.0, 0.0, 1.0)
    
    return (nx / length, ny / length, nz / length)

def get_color_from_height(height):
    if height < 0.2:
        return (0.0, 0.0, 0.5)  # Deep water (normalized to 0.0-1.0)
    elif height < 0.4:
        return (0.0, 0.0, 1.0)  # Shallow water
    elif height < 0.5:
        return (0.94, 0.94, 0.25)  # Beach (240/255, 240/255, 64/255)
    elif height < 0.7:
        return (0.13, 0.55, 0.13)  # Lowlands (34/255, 139/255, 34/255)
    elif height < 0.85:
        return (0.55, 0.27, 0.07)  # Mountains (139/255, 69/255, 19/255)
    else:
        return (1.0, 1.0, 1.0)  # Snow caps

def generate_mash(map):
    triangles = []
    height_multiplier = HEIGHT_MULTIPLIER
    size_x, size_y = map.shape
    for i in range(size_x - 1):
        for j in range(size_y - 1):
            v1 = (i, j, map[i][j])
            v2 = (i + 1, j, map[i + 1][j])
            v3 = (i, j + 1, map[i][j + 1])
            v4 = (i + 1, j + 1, map[i + 1][j + 1])

            v1 = (v1[0], v1[1], v1[2] * height_multiplier)
            v2 = (v2[0], v2[1], v2[2] * height_multiplier)
            v3 = (v3[0], v3[1], v3[2] * height_multiplier)
            v4 = (v4[0], v4[1], v4[2] * height_multiplier)
            
            normal1 = compute_normal(v1, v2, v3)
            color1 = get_color_from_height((v1[2] + v2[2] + v3[2]) / 3)
            triangle1 = Triangle(v1[0], v1[1], v1[2],
                                 v2[0], v2[1], v2[2],
                                 v3[0], v3[1], v3[2],
                                 normal1[0], normal1[1], normal1[2],
                                 color1[0], color1[1], color1[2])
            triangles.append(triangle1)
            
            normal2 = compute_normal(v2, v4, v3)
            color2 = get_color_from_height((v2[2] + v4[2] + v3[2]) / 3)
            triangle2 = Triangle(v2[0], v2[1], v2[2],
                                 v4[0], v4[1], v4[2],
                                 v3[0], v3[1], v3[2],
                                 normal2[0], normal2[1], normal2[2],
                                 color2[0], color2[1], color2[2])
            triangles.append(triangle2)

    numOfTriangles = len(triangles)

    print(f"Generated {numOfTriangles} triangles for the mesh.")

    triangleStructSize = 76
    fileSize = 8 + numOfTriangles * triangleStructSize

    with open("terrain.bin", "wb") as f:
        # write uint32 file size
        f.write(fileSize.to_bytes(4, byteorder='little'))
        # write uint32 triangle struct size (to match Go format)
        f.write(triangleStructSize.to_bytes(4, byteorder='little'))

        for i, triangle in enumerate(triangles):
            # Vertices (36 bytes: 9 float32s)
            f.write(np.float32(triangle.v1x).tobytes())
            f.write(np.float32(triangle.v1y).tobytes())
            f.write(np.float32(triangle.v1z).tobytes())
            f.write(np.float32(triangle.v2x).tobytes())
            f.write(np.float32(triangle.v2y).tobytes())
            f.write(np.float32(triangle.v2z).tobytes())
            f.write(np.float32(triangle.v3x).tobytes())
            f.write(np.float32(triangle.v3y).tobytes())
            f.write(np.float32(triangle.v3z).tobytes())
            # Normal (12 bytes: 3 float32s)
            f.write(np.float32(triangle.normalx).tobytes())
            f.write(np.float32(triangle.normaly).tobytes())
            f.write(np.float32(triangle.normalz).tobytes())
            # Material properties (12 bytes: 3 float32s)
            f.write(np.float32(triangle.roughness).tobytes())
            f.write(np.float32(triangle.metallic).tobytes())
            f.write(np.float32(triangle.emission).tobytes())
            # Color as float32 to match Go format (12 bytes: 3 float32s)
            f.write(np.float32(triangle.colorR).tobytes())
            f.write(np.float32(triangle.colorG).tobytes())
            f.write(np.float32(triangle.colorB).tobytes())
            # Triangle index (4 bytes: 1 uint32)
            f.write(i.to_bytes(4, byteorder='little'))

    # check if file size is correct
    actualFileSize = 0
    with open("terrain.bin", "rb") as f:
        f.seek(0, 2)  # move to end of file
        actualFileSize = f.tell()
    if actualFileSize == fileSize:
        print(f"Binary mesh file 'terrain.bin' generated successfully with size {actualFileSize} bytes.")
        print(f"Header: 8 bytes (fileSize + triangleStructSize)")
        print(f"Triangles: {numOfTriangles} × {triangleStructSize} = {numOfTriangles * triangleStructSize} bytes")
        print(f"Total: {fileSize} bytes")
    else:
        print(f"Error in file size: expected {fileSize}, got {actualFileSize}.")
            


if __name__ == "__main__":
    test_map = generate_enhanced_height_map(MAP_SIZE, seed=42)
    print("Shape of generated map:", test_map.shape)
    
    print("\nGenerating binary mesh file...")
    generate_mash(test_map)


    print(f"Generated {MAP_SIZE}x{MAP_SIZE} enhanced terrain map")
    
    display_height_map(test_map, 
                      title=f"Enhanced Terrain Map ({MAP_SIZE}x{MAP_SIZE})",
                      save_path="enhanced_terrain_map.png")
    
    print("Generating 3D terrain view...")
    generate_3d_view(test_map, 
                    title=f"3D Terrain View ({MAP_SIZE}x{MAP_SIZE})",
                    save_path="terrain_3d_view.png")
