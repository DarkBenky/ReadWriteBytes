from PIL import Image
import os

CHAR_SIZE = 8
path = "01.png"
out_dir = "chars"
os.makedirs(out_dir, exist_ok=True)

img = Image.open(path).convert("L")
w, h = img.size
cols = w // CHAR_SIZE

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
        print(f"Saved {filename} (ASCII {ascii_code}, '{ch if 32<=ascii_code<=126 else ''}')")
