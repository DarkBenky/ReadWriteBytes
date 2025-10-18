from PIL import Image
import os
import struct

CHAR_SIZE = 8
path = "01.png"
out_dir = "chars"
os.makedirs(out_dir, exist_ok=True)

img = Image.open(path).convert("L")
w, h = img.size
cols = w // CHAR_SIZE

def convert_to_bitArray(image):
    """Convert an image to a bit array representation."""
    pixels = image.load()
    bit_array = []
    for y in range(image.height):
        row = 0
        for x in range(image.width):
            if pixels[x, y] > 128:
                bit_array.append(1)  # Black pixel
            else:
                bit_array.append(0)
    width, height = image.size
    return bit_array, width, height

def construct_image_from_bits(bits, width, height):
    """Construct an image from a bit array."""
    img = Image.new("L", (width, height))
    pixels = img.load()
    
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            pixels[x, y] = 0 if bits[idx] else 255  # Black or white pixel
    
    return img

bits, width, height = convert_to_bitArray(img)

# save bits as a binary file with width and height as uint32
with open("fonts.bin", "wb") as f:
    # Write width and height as uint32 (4 bytes each, little-endian)
    f.write(struct.pack('<I', width))   # uint32 little-endian
    f.write(struct.pack('<I', height))  # uint32 little-endian

    print(f"Saving {len(bits)} bits to fonts.bin with dimensions {width}x{height}")
    
    # Write the bit data
    for bit in bits:
        f.write(bytes([bit]))

# Construct an image from the bit array
img = construct_image_from_bits(bits, w, h)
# show image
img.save("output.png")




for y in range(0, h, CHAR_SIZE):
    for x in range(0, w, CHAR_SIZE):
        # Calculate ASCII code (starting at 32)
        idx = (y // CHAR_SIZE) * cols + (x // CHAR_SIZE)
        ascii_code = 32 + idx

        # Only include printable ASCII range (32–126) or extended if needed
        if 32 <= ascii_code <= 126:
            ch = chr(ascii_code)
            safe_name = ch if ch.isalnum() else f"0x{ascii_code:02X}"
        else:
            # Use hex for non-printable or extended chars
            safe_name = f"0x{ascii_code:02X}"

        # Crop and scale
        char_img = img.crop((x, y, x + CHAR_SIZE, y + CHAR_SIZE))
        char_img = char_img.resize((CHAR_SIZE * 8, CHAR_SIZE * 8), Image.NEAREST)

        filename = f"{out_dir}/{safe_name}.png"
        char_img.save(filename)
        # print(f"Saved {filename} (ASCII {ascii_code}, '{ch if 32<=ascii_code<=126 else ''}')")
