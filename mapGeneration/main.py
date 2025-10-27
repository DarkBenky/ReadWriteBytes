from noise import pnoise2
import numpy as np
import random
from PIL import Image

MAP_SIZE = 450 # 15 km
ITERATIONS = 3
HEIGHT_RANGE = (-45, 22.5)  # Elevation range with more dramatic variation

WATER_MERGE_THRESHOLD = 0.02
FLAT_TERRAIN_MERGE_THRESHOLD = 0.0125
MOUNTAIN_MERGE_THRESHOLD = 0.05

# Material properties for different terrain types
class MaterialProperties:
    def __init__(self, roughness, metallic, emission):
        self.roughness = roughness
        self.metallic = metallic
        self.emission = emission

MATERIALS = {
    'deep_water': MaterialProperties(roughness=0.05, metallic=0.15, emission=0.0),
    'shallow_water': MaterialProperties(roughness=0.1, metallic=0.10, emission=0.0),
    'beach': MaterialProperties(roughness=0.85, metallic=0.0, emission=0.0),
    'grass': MaterialProperties(roughness=0.9, metallic=0.0, emission=0.0),
    'rock': MaterialProperties(roughness=0.7, metallic=0.02, emission=0.0),
    'mountain': MaterialProperties(roughness=0.6, metallic=0.02, emission=0.0),
    'snow': MaterialProperties(roughness=0.3, metallic=0.01, emission=0.1)
}

def generate_height_map(size=MAP_SIZE, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create base terrain with multiple octaves of Perlin noise
    height_map = np.zeros((size, size))
    
    # More dramatic octave parameters for interesting terrain
    octave_params = [
        (0.5, 120.0, 1.2),    # Large scale features (continents/valleys) - bigger amplitude
        (0.3, 60.0, 0.7),     # Medium scale features (hills) - more pronounced
        (0.18, 30.0, 0.4),    # Small scale features (bumps)
        (0.1, 15.0, 0.2),     # Fine detail
        (0.06, 8.0, 0.1),     # Very fine detail
        (0.03, 4.0, 0.05),    # Ultra fine detail for texture
    ]
    
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    
    for scale, frequency, amplitude_factor in octave_params:
        amplitude = (HEIGHT_RANGE[1] - HEIGHT_RANGE[0]) * amplitude_factor
        
        noise = np.vectorize(lambda i, j: pnoise2(
            i / frequency, 
            j / frequency, 
            octaves=6, 
            persistence=0.55,
            lacunarity=2.3,
            repeatx=size, 
            repeaty=size, 
            base=seed or 0
        ))(xx, yy)
        
        height_map += noise * amplitude
    
    # Add dramatic plateaus
    plateau_noise = np.vectorize(lambda i, j: pnoise2(
        i / 90.0, 
        j / 90.0, 
        octaves=3, 
        base=(seed or 0) + 5000
    ))(xx, yy)
    plateau_mask = plateau_noise > 0.3
    height_map[plateau_mask] += 25.0
    
    # Add deep valleys with more variation
    valley_noise = np.vectorize(lambda i, j: pnoise2(
        i / 70.0, 
        j / 70.0, 
        octaves=4, 
        base=(seed or 0) + 3000
    ))(xx, yy)
    valley_mask = valley_noise < -0.35
    height_map[valley_mask] -= 20.0
    
    # Apply hydraulic-style erosion for realistic valleys
    from scipy.ndimage import gaussian_filter
    height_map = gaussian_filter(height_map, sigma=1.2)
    
    # Apply thermal erosion to smooth steep slopes
    height_map = apply_thermal_erosion(height_map, iterations=2, talus_angle=0.12)
    
    # Add dramatic ridges for mountain ranges
    ridge_noise = np.vectorize(lambda i, j: abs(pnoise2(
        i / 65.0, 
        j / 65.0, 
        octaves=5, 
        base=(seed or 0) + 1000
    )))(xx, yy)
    
    # Create mountain peaks with exponential scaling
    peak_noise = np.vectorize(lambda i, j: pnoise2(
        i / 50.0, 
        j / 50.0, 
        octaves=4, 
        base=(seed or 0) + 2000
    ))(xx, yy)
    
    # Only add ridges to higher elevations
    ridge_mask = height_map > (HEIGHT_RANGE[1] - HEIGHT_RANGE[0]) * 0.25
    height_map[ridge_mask] += ridge_noise[ridge_mask] * 22.0
    
    # Add sharp peaks
    peak_mask = (height_map > (HEIGHT_RANGE[1] - HEIGHT_RANGE[0]) * 0.5) & (peak_noise > 0.4)
    height_map[peak_mask] += 18.0
    
    # Create interesting coastal features with islands
    coastal_noise = np.vectorize(lambda i, j: pnoise2(
        i / 40.0, 
        j / 40.0, 
        octaves=5, 
        base=(seed or 0) + 4000
    ))(xx, yy)
    
    # Normalize to HEIGHT_RANGE
    min_val = np.min(height_map)
    max_val = np.max(height_map)
    if max_val > min_val:
        height_map = (height_map - min_val) / (max_val - min_val)
        height_map = height_map * (HEIGHT_RANGE[1] - HEIGHT_RANGE[0]) + HEIGHT_RANGE[0]
    
    # Clip to minimum height (sea level can be below 0)
    height_map = np.clip(height_map, HEIGHT_RANGE[0], HEIGHT_RANGE[1])
    
    # Create more interesting water features - islands and varied coastlines
    min_h, max_h = HEIGHT_RANGE
    range_h = max_h - min_h
    water_threshold = min_h + range_h * 0.28  # Shallow water threshold
    sea_level = min_h + range_h * 0.27  # Set sea level just below beach threshold
    
    # Add small islands by raising some water areas
    island_mask = (height_map < water_threshold) & (coastal_noise > 0.6)
    height_map[island_mask] = min_h + range_h * 0.40  # Raise to beach/grass level
    
    # Set remaining water areas to flat surface at sea level
    water_mask = height_map < water_threshold
    height_map[water_mask] = sea_level
    
    return height_map


def apply_thermal_erosion(height_map, iterations=1, talus_angle=0.08):
    """Apply thermal erosion to smooth steep slopes"""
    for _ in range(iterations):
        # Create shifted versions of the array for all 8 neighbors
        neighbors = [
            np.roll(height_map, 1, axis=0),   # up
            np.roll(height_map, -1, axis=0),  # down
            np.roll(height_map, 1, axis=1),   # left
            np.roll(height_map, -1, axis=1),  # right
            np.roll(np.roll(height_map, 1, axis=0), 1, axis=1),    # up-left
            np.roll(np.roll(height_map, 1, axis=0), -1, axis=1),   # up-right
            np.roll(np.roll(height_map, -1, axis=0), 1, axis=1),   # down-left
            np.roll(np.roll(height_map, -1, axis=0), -1, axis=1),  # down-right
        ]
        
        # Find maximum difference with any neighbor
        max_diff = np.zeros_like(height_map)
        for neighbor in neighbors:
            diff = height_map - neighbor
            max_diff = np.maximum(max_diff, diff)
        
        # Erode where slope exceeds talus angle
        erosion_mask = max_diff > talus_angle
        height_map[erosion_mask] -= (max_diff[erosion_mask] - talus_angle) * 0.5
        
        # Fix edges that were affected by roll
        height_map[0, :] = height_map[1, :]
        height_map[-1, :] = height_map[-2, :]
        height_map[:, 0] = height_map[:, 1]
        height_map[:, -1] = height_map[:, -2]
    
    return height_map

def get_color_from_height_rgb(height):
    """Dynamic color mapping based on actual HEIGHT_RANGE"""
    min_h, max_h = HEIGHT_RANGE
    range_h = max_h - min_h
    
    # Define thresholds as percentages of the actual height range
    deep_water = min_h + range_h * 0.15      # Deep water
    shallow_water = min_h + range_h * 0.30   # Shallow water
    beach = min_h + range_h * 0.35           # Beach/shore
    grass = min_h + range_h * 0.55           # Grass/lowlands
    rock = min_h + range_h * 0.70            # Rocky terrain
    mountain = min_h + range_h * 0.85        # Mountains
    # Above mountain = snow
    
    if height < deep_water:
        return (0, 17, 51)           # Deep water (dark blue)
    elif height < shallow_water:
        return (0, 51, 102)          # Shallow water (blue)
    elif height < beach:
        return (240, 240, 64)        # Beach (sandy yellow)
    elif height < grass:
        return (34, 139, 34)         # Grass (forest green)
    elif height < rock:
        return (139, 69, 19)         # Rock (saddle brown)
    elif height < mountain:
        return (160, 82, 45)         # Mountain (sienna)
    else:
        return (255, 255, 255)       # Snow (white)

def save_height_map_image(height_map, save_path="terrain_map.png"):
    height_array = np.array(height_map)
    size_x, size_y = height_array.shape
    
    img = Image.new('RGB', (size_y, size_x))
    pixels = img.load()
    
    for i in range(size_x):
        for j in range(size_y):
            color = get_color_from_height_rgb(height_array[i][j])
            pixels[j, i] = color
    
    img.save(save_path)
    print(f"Terrain map saved to {save_path}")
    
    stats = f"""Map Statistics:
Min Elevation: {np.min(height_array):.3f}
Max Elevation: {np.max(height_array):.3f}
Mean Elevation: {np.mean(height_array):.3f}
Std Deviation: {np.std(height_array):.3f}"""
    print(stats)

def get_height_color_gradient(height):
    height = np.clip(height, 0.0, 1.0)
    
    if height < 0.25:
        t = height / 0.25
        r = int(0 * (1-t) + 0 * t)
        g = int(0 * (1-t) + 100 * t)
        b = int(139 * (1-t) + 255 * t)
    elif height < 0.5:
        t = (height - 0.25) / 0.25
        r = int(0 * (1-t) + 0 * t)
        g = int(100 * (1-t) + 255 * t)
        b = int(255 * (1-t) + 255 * t)
    elif height < 0.75:
        t = (height - 0.5) / 0.25
        r = int(0 * (1-t) + 255 * t)
        g = int(255 * (1-t) + 255 * t)
        b = int(255 * (1-t) + 0 * t)
    else:
        t = (height - 0.75) / 0.25
        r = int(255 * (1-t) + 255 * t)
        g = int(255 * (1-t) + 100 * t)
        b = int(0 * (1-t) + 0 * t)
    
    return (r, g, b)

def save_height_gradient_image(height_map, save_path="height_gradient.png"):
    height_array = np.array(height_map)
    min_h = np.min(height_array)
    max_h = np.max(height_array)
    
    normalized = (height_array - min_h) / (max_h - min_h) if max_h > min_h else height_array
    
    size_x, size_y = height_array.shape
    img = Image.new('RGB', (size_y, size_x))
    pixels = img.load()
    
    for i in range(size_x):
        for j in range(size_y):
            color = get_height_color_gradient(normalized[i][j])
            pixels[j, i] = color
    
    img.save(save_path)
    print(f"Height gradient map saved to {save_path}")


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
    """Dynamic color mapping for mesh generation based on HEIGHT_RANGE"""
    min_h, max_h = HEIGHT_RANGE
    range_h = max_h - min_h
    
    # Define thresholds as percentages of the actual height range
    deep_water = min_h + range_h * 0.15
    shallow_water = min_h + range_h * 0.30
    beach = min_h + range_h * 0.35
    grass = min_h + range_h * 0.55
    rock = min_h + range_h * 0.70
    mountain = min_h + range_h * 0.85
    
    if height < deep_water:
        return (0.0, 0.07, 0.2)      # Deep water (dark blue)
    elif height < shallow_water:
        return (0.0, 0.2, 0.4)       # Shallow water (blue)
    elif height < beach:
        return (0.94, 0.94, 0.25)    # Beach (sandy)
    elif height < grass:
        return (0.13, 0.55, 0.13)    # Grass (green)
    elif height < rock:
        return (0.55, 0.27, 0.07)    # Rock (brown)
    elif height < mountain:
        return (0.63, 0.32, 0.18)    # Mountain (lighter brown)
    else:
        return (1.0, 1.0, 1.0)       # Snow (white)

def get_material_from_height(height):
    """Dynamic material assignment based on HEIGHT_RANGE"""
    min_h, max_h = HEIGHT_RANGE
    range_h = max_h - min_h
    
    deep_water = min_h + range_h * 0.15
    shallow_water = min_h + range_h * 0.30
    beach = min_h + range_h * 0.35
    grass = min_h + range_h * 0.55
    rock = min_h + range_h * 0.70
    mountain = min_h + range_h * 0.85
    
    if height < deep_water:
        return MATERIALS['deep_water']
    elif height < shallow_water:
        return MATERIALS['shallow_water']
    elif height < beach:
        return MATERIALS['beach']
    elif height < grass:
        return MATERIALS['grass']
    elif height < rock:
        return MATERIALS['rock']
    elif height < mountain:
        return MATERIALS['mountain']
    else:
        return MATERIALS['snow']

def is_water(height):
    """Check if height is water based on HEIGHT_RANGE"""
    min_h, max_h = HEIGHT_RANGE
    range_h = max_h - min_h
    shallow_water_threshold = min_h + range_h * 0.30
    return height < shallow_water_threshold

def get_terrain_type(height):
    """Get terrain type ID based on HEIGHT_RANGE"""
    min_h, max_h = HEIGHT_RANGE
    range_h = max_h - min_h
    
    deep_water = min_h + range_h * 0.15
    shallow_water = min_h + range_h * 0.30
    beach = min_h + range_h * 0.35
    grass = min_h + range_h * 0.55
    rock = min_h + range_h * 0.70
    mountain = min_h + range_h * 0.85
    
    if height < deep_water:
        return 0
    elif height < shallow_water:
        return 1
    elif height < beach:
        return 2
    elif height < grass:
        return 3
    elif height < rock:
        return 4
    elif height < mountain:
        return 5
    else:
        return 6

def can_merge_quads(map, i1, j1, i2, j2, height_multiplier):
    h1 = map[i1][j1]
    h2 = map[i1+1][j1]
    h3 = map[i1][j1+1]
    h4 = map[i1+1][j1+1]
    
    h5 = map[i2][j2]
    h6 = map[i2+1][j2]
    h7 = map[i2][j2+1]
    h8 = map[i2+1][j2+1]
    
    type1 = get_terrain_type(h1)
    type5 = get_terrain_type(h5)
    
    if type1 != type5:
        return False
    
    if is_water(h1):
        return (abs(h1 - h5) < WATER_MERGE_THRESHOLD and 
                abs(h2 - h6) < WATER_MERGE_THRESHOLD and 
                abs(h3 - h7) < WATER_MERGE_THRESHOLD and 
                abs(h4 - h8) < WATER_MERGE_THRESHOLD)
    else:
        height_threshold = FLAT_TERRAIN_MERGE_THRESHOLD if type1 <= 3 else MOUNTAIN_MERGE_THRESHOLD
        return (abs(h1 - h5) < height_threshold and abs(h2 - h6) < height_threshold and 
                abs(h3 - h7) < height_threshold and abs(h4 - h8) < height_threshold)

def generate_mash(map):
    triangles = []
    size_x, size_y = map.shape
    
    processed = np.zeros((size_x, size_y), dtype=bool)
    
    print("Applying greedy meshing optimization...")
    for i in range(size_x - 1):
        for j in range(size_y - 1):
            if processed[i][j]:
                continue
            
            v1 = (i, j, map[i][j])
            v2 = (i + 1, j, map[i + 1][j] )
            v3 = (i, j + 1, map[i][j + 1])
            v4 = (i + 1, j + 1, map[i + 1][j + 1])
            
            avg_height = (v1[2] + v2[2] + v3[2] + v4[2]) / 4
            terrain_type = get_terrain_type(avg_height)
            
            width = 1
            height = 1
            
            while j + width < size_y - 1:
                can_extend = True
                for ii in range(i, i + height):
                    if ii >= size_x - 1 or processed[ii][j + width]:
                        can_extend = False
                        break
                    if not can_merge_quads(map, i, j, ii, j + width, 1):
                        can_extend = False
                        break
                if can_extend:
                    width += 1
                else:
                    break
            
            while i + height < size_x - 1:
                can_extend = True
                for jj in range(j, j + width):
                    if jj >= size_y - 1 or processed[i + height][jj]:
                        can_extend = False
                        break
                    if not can_merge_quads(map, i, j, i + height, jj, 1):
                        can_extend = False
                        break
                if can_extend:
                    height += 1
                else:
                    break
            
            for ii in range(i, min(i + height, size_x - 1)):
                for jj in range(j, min(j + width, size_y - 1)):
                    processed[ii][jj] = True
            
            v1_merged = (i, j, map[i][j])
            v2_merged = (i + height, j, map[min(i + height, size_x - 1)][j])
            v3_merged = (i, j + width, map[i][min(j + width, size_y - 1)])
            v4_merged = (i + height, j + width, map[min(i + height, size_x - 1)][min(j + width, size_y - 1)])
            
            normal = compute_normal(v1_merged, v2_merged, v3_merged)
            base_color = get_color_from_height(avg_height)
            
            # Add color variation
            color_variation = random.uniform(-0.08, 0.08)
            color = (
                np.clip(base_color[0] + color_variation, 0.0, 1.0),
                np.clip(base_color[1] + color_variation, 0.0, 1.0),
                np.clip(base_color[2] + color_variation, 0.0, 1.0)
            )
            
            material = get_material_from_height(avg_height)
            
            triangle1 = Triangle(v1_merged[0], v1_merged[1], v1_merged[2],
                                 v2_merged[0], v2_merged[1], v2_merged[2],
                                 v3_merged[0], v3_merged[1], v3_merged[2],
                                 normal[0], normal[1], normal[2],
                                 color[0], color[1], color[2])
            triangle1.roughness = material.roughness
            triangle1.metallic = material.metallic
            triangle1.emission = material.emission
            triangles.append(triangle1)
            
            triangle2 = Triangle(v2_merged[0], v2_merged[1], v2_merged[2],
                                 v4_merged[0], v4_merged[1], v4_merged[2],
                                 v3_merged[0], v3_merged[1], v3_merged[2],
                                 normal[0], normal[1], normal[2],
                                 color[0], color[1], color[2])
            triangle2.roughness = material.roughness
            triangle2.metallic = material.metallic
            triangle2.emission = material.emission
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
    test_map = generate_height_map(MAP_SIZE, seed=421)
    print("Shape of generated map:", test_map.shape)
    
    print("\nGenerating binary mesh file...")
    generate_mash(test_map)
    
    print(f"\nGenerated {MAP_SIZE}x{MAP_SIZE} enhanced terrain map")
    save_height_map_image(test_map, save_path="terrain_map.png")
    save_height_gradient_image(test_map, save_path="height_gradient.png")
