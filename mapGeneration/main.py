from noise import pnoise2
import numpy as np
import random
from PIL import Image

MAP_SIZE = 2_000 # 15 km
ITERATIONS = 3
HEIGHT_MULTIPLIER = 0.45

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
    'deep_water': MaterialProperties(roughness=0.05, metallic=0.85, emission=0.0),
    'shallow_water': MaterialProperties(roughness=0.1, metallic=0.75, emission=0.0),
    'beach': MaterialProperties(roughness=0.85, metallic=0.0, emission=0.0),
    'grass': MaterialProperties(roughness=0.9, metallic=0.0, emission=0.0),
    'rock': MaterialProperties(roughness=0.7, metallic=0.1, emission=0.0),
    'mountain': MaterialProperties(roughness=0.6, metallic=0.15, emission=0.0),
    'snow': MaterialProperties(roughness=0.3, metallic=0.05, emission=0.1)
}

def generate_enhanced_height_map(size=MAP_SIZE, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    height_map = np.zeros((size, size))
    center_x, center_y = size / 2, size / 2
    max_distance = np.sqrt(2) * (size / 2)
    
    print("Generating radial base with center landmass...")
    for i in range(size):
        for j in range(size):
            dx = i - center_x
            dy = j - center_y
            distance = np.sqrt(dx*dx + dy*dy)
            norm_distance = distance / max_distance
            
            radial_height = np.exp(-norm_distance * 3.0) * 1.8 - 0.3
            height_map[i][j] = radial_height
    
    print("Adding continental noise patterns...")
    frequency = 0.006
    amplitude = 0.5
    persistence = 0.5
    lacunarity = 2.2
    
    for i in range(size):
        for j in range(size):
            noise_value = 0
            freq = frequency
            amp = amplitude
            
            for octave in range(6):
                noise_value += pnoise2(i * freq + 500, j * freq + 500, octaves=1) * amp
                freq *= lacunarity
                amp *= persistence
            
            height_map[i][j] += noise_value * 0.8
    
    print("Creating islands in ocean areas...")
    num_islands = random.randint(15, 25)
    for _ in range(num_islands):
        angle = random.uniform(0, 2 * np.pi)
        island_distance = random.uniform(0.5, 0.95) * max_distance
        island_x = int(center_x + np.cos(angle) * island_distance)
        island_y = int(center_y + np.sin(angle) * island_distance)
        island_size = random.uniform(20, 60)
        island_height = random.uniform(0.4, 0.8)
        
        for i in range(size):
            for j in range(size):
                dx = i - island_x
                dy = j - island_y
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < island_size:
                    island_influence = np.exp(-(dist / island_size) ** 2) * island_height
                    height_map[i][j] += island_influence
    
    print("Adding mountain ridges to central landmass...")
    ridge_frequency = 0.003
    for i in range(size):
        for j in range(size):
            if height_map[i][j] > 0.15:
                ridge_noise = abs(pnoise2(i * ridge_frequency + 2000, j * ridge_frequency + 2000, octaves=1))
                ridge_height = 1.0 - ridge_noise
                ridge_height = ridge_height ** 2.5
                height_map[i][j] += ridge_height * 0.35
    
    print("Carving river valleys...")
    valley_frequency = 0.006
    for i in range(size):
        for j in range(size):
            if height_map[i][j] > 0.1:  # Only carve valleys on land
                valley_noise = abs(pnoise2(i * valley_frequency + 1000, j * valley_frequency + 1000, octaves=1))
                valley_depth = (1.0 - valley_noise) ** 3
                height_map[i][j] -= valley_depth * 0.2
    
    print("Adding surface details...")
    detail_frequency = 0.03
    for i in range(size):
        for j in range(size):
            detail = pnoise2(i * detail_frequency, j * detail_frequency, octaves=2)
            height_map[i][j] += detail * 0.05
    
    print("Applying erosion effects...")
    height_map = apply_thermal_erosion(height_map, iterations=ITERATIONS, talus_angle=0.12)
    
    height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
    height_map = height_map * 1.4 - 0.4
    
    print("Flattening water areas...")
    water_level = 0.4
    for i in range(size):
        for j in range(size):
            if height_map[i][j] < water_level:
                height_map[i][j] = water_level * 0.5
    
    height_map = np.clip(height_map, 0.0, None)
    
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

def get_color_from_height_rgb(height):
    if height < 0.2:
        return (0, 17, 51)
    elif height < 0.4:
        return (0, 51, 102)
    elif height < 0.5:
        return (240, 240, 64)
    elif height < 0.7:
        return (34, 139, 34)
    elif height < 0.85:
        return (139, 69, 19)
    elif height < 0.95:
        return (160, 82, 45)
    else:
        return (255, 255, 255)

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

def get_material_from_height(height):
    if height < 0.2:
        return MATERIALS['deep_water']
    elif height < 0.4:
        return MATERIALS['shallow_water']
    elif height < 0.5:
        return MATERIALS['beach']
    elif height < 0.7:
        return MATERIALS['grass']
    elif height < 0.85:
        return MATERIALS['rock']
    elif height < 0.95:
        return MATERIALS['mountain']
    else:
        return MATERIALS['snow']

def is_water(height):
    return height < 0.4

def get_terrain_type(height):
    if height < 0.2:
        return 0
    elif height < 0.4:
        return 1
    elif height < 0.5:
        return 2
    elif height < 0.7:
        return 3
    elif height < 0.85:
        return 4
    elif height < 0.95:
        return 5
    else:
        return 6

def can_merge_quads(map, i1, j1, i2, j2, height_multiplier):
    h1 = map[i1][j1] * height_multiplier
    h2 = map[i1+1][j1] * height_multiplier
    h3 = map[i1][j1+1] * height_multiplier
    h4 = map[i1+1][j1+1] * height_multiplier
    
    h5 = map[i2][j2] * height_multiplier
    h6 = map[i2+1][j2] * height_multiplier
    h7 = map[i2][j2+1] * height_multiplier
    h8 = map[i2+1][j2+1] * height_multiplier
    
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
    height_multiplier = HEIGHT_MULTIPLIER
    size_x, size_y = map.shape
    
    processed = np.zeros((size_x, size_y), dtype=bool)
    
    print("Applying greedy meshing optimization...")
    for i in range(size_x - 1):
        for j in range(size_y - 1):
            if processed[i][j]:
                continue
            
            v1 = (i, j, map[i][j] * height_multiplier)
            v2 = (i + 1, j, map[i + 1][j] * height_multiplier)
            v3 = (i, j + 1, map[i][j + 1] * height_multiplier)
            v4 = (i + 1, j + 1, map[i + 1][j + 1] * height_multiplier)
            
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
                    if not can_merge_quads(map, i, j, ii, j + width, height_multiplier):
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
                    if not can_merge_quads(map, i, j, i + height, jj, height_multiplier):
                        can_extend = False
                        break
                if can_extend:
                    height += 1
                else:
                    break
            
            for ii in range(i, min(i + height, size_x - 1)):
                for jj in range(j, min(j + width, size_y - 1)):
                    processed[ii][jj] = True
            
            v1_merged = (i, j, map[i][j] * height_multiplier)
            v2_merged = (i + height, j, map[min(i + height, size_x - 1)][j] * height_multiplier)
            v3_merged = (i, j + width, map[i][min(j + width, size_y - 1)] * height_multiplier)
            v4_merged = (i + height, j + width, map[min(i + height, size_x - 1)][min(j + width, size_y - 1)] * height_multiplier)
            
            normal = compute_normal(v1_merged, v2_merged, v3_merged)
            color = get_color_from_height(avg_height)
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
    test_map = generate_enhanced_height_map(MAP_SIZE, seed=42)
    print("Shape of generated map:", test_map.shape)
    
    print("\nGenerating binary mesh file...")
    generate_mash(test_map)
    
    print(f"\nGenerated {MAP_SIZE}x{MAP_SIZE} enhanced terrain map")
    save_height_map_image(test_map, save_path="terrain_map.png")
    save_height_gradient_image(test_map, save_path="height_gradient.png")
