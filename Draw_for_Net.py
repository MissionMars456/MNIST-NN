import tkinter as tk
import numpy as np
from scipy.ndimage import center_of_mass, shift
import Neural_Net as nn

root = tk.Tk()
root.title("Input Drawer")
root.geometry("560x600")

canvas = tk.Canvas(root, width=560, height=560, bg="black")
canvas.pack()

prediction_label = tk.Label(root, text="Prediction: Draw something...", font=("Arial", 16, "bold"))
prediction_label.pack(pady=10)

SIZE = 20
ROWS = 560 // SIZE 
COLS = 560 // SIZE 

grid_items = {}
grid_values = {}
last_brushed_cell = None

def create_board():
    """Initializes or resets the board tiles to pure black."""
    global grid_items, grid_values
    
    canvas.delete("all")
    grid_items.clear()
    grid_values.clear()
    
    for r in range(ROWS):
        for c in range(COLS):
            x1, y1 = c * SIZE, r * SIZE
            x2, y2 = x1 + SIZE, y1 + SIZE
            
            rect_id = canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="#2B2B2B")
            grid_items[(r, c)] = rect_id
            grid_values[(r, c)] = 0.0

def change_intensity(r, c, amount):
    """Changes a tile's brightness value and updates its color on screen."""
    if (r, c) in grid_values:
        grid_values[(r, c)] = max(0.0, min(1.0, grid_values[(r, c)] + amount))
        
        color_byte = int(grid_values[(r, c)] * 255)
        color_hex = f"#{color_byte:02x}{color_byte:02x}{color_byte:02x}"
        canvas.itemconfig(grid_items[(r, c)], fill=color_hex)

def apply_brush(event, is_drawing, is_initial_click=False):
    """Applies unequal weights to the center, edges, and corners of a 3x3 area."""
    global last_brushed_cell
    
    c = event.x // SIZE
    r = event.y // SIZE
    current_cell = (r, c)
    
    if not is_initial_click and current_cell == last_brushed_cell:
        return
        
    last_brushed_cell = current_cell
    
    center_weight = 0.8 if is_drawing else -0.8
    edge_weight   = 0.4 if is_drawing else -0.4
    corner_weight = 0.15 if is_drawing else -0.15

    # 1. Center Tile
    change_intensity(r, c, center_weight)
    
    # 2. Four Edge Neighbors
    change_intensity(r - 1, c, edge_weight)
    change_intensity(r + 1, c, edge_weight)
    change_intensity(r, c - 1, edge_weight)
    change_intensity(r, c + 1, edge_weight)
    
    # 3. Four Corner Neighbors
    change_intensity(r - 1, c - 1, corner_weight)
    change_intensity(r - 1, c + 1, corner_weight)
    change_intensity(r + 1, c - 1, corner_weight)
    change_intensity(r + 1, c + 1, corner_weight)

def preprocess_drawing(grid_values):
    """Centers the drawing via center of mass to align with MNIST formatting."""
    img_2d = np.zeros((28, 28))
    for (r, c), val in grid_values.items():
        img_2d[r, c] = val
        
    if np.sum(img_2d) == 0:
        return np.zeros((1, 784))
        
    cy, cx = center_of_mass(img_2d)
    rows, cols = img_2d.shape
    shiftx = (cols / 2.0) - cx
    shifty = (rows / 2.0) - cy
    
    shifted_img = shift(img_2d, [shifty, shiftx], cval=0.0)
    return shifted_img.reshape(1, 784)

def process_and_predict():
    """Extracts data, applies centering, and runs the neural network model."""
    try:
        img = preprocess_drawing(grid_values)
        prediction_matrix = nn.runNet(img, None)
        
        predicted_digit = int(np.argmax(prediction_matrix))
        confidence = float(np.max(prediction_matrix)) * 100
        
        prediction_label.config(text=f"Prediction: {predicted_digit} ({confidence:.1f}%)")
    except Exception as e:
        prediction_label.config(text=f"Network Error: {str(e)}")

def reset_drag_tracking(event):
    global last_brushed_cell
    last_brushed_cell = None
    process_and_predict()

def clear_all(event):
    global last_brushed_cell
    last_brushed_cell = None
    create_board()
    prediction_label.config(text="Prediction: Draw something...")

# Generate the initial board
create_board()

# Left Click & Drag to Draw
canvas.bind("<Button-1>", lambda e: apply_brush(e, is_drawing=True, is_initial_click=True))
canvas.bind("<B1-Motion>", lambda e: apply_brush(e, is_drawing=True, is_initial_click=False))

# Right Click & Drag to Erase
canvas.bind("<Button-3>", lambda e: apply_brush(e, is_drawing=False, is_initial_click=True))
canvas.bind("<B3-Motion>", lambda e: apply_brush(e, is_drawing=False, is_initial_click=False))

# Middle Click to Clear
canvas.bind("<Button-2>", clear_all)

# Handle button releases to trigger predictions
canvas.bind("<ButtonRelease-1>", reset_drag_tracking)
canvas.bind("<ButtonRelease-3>", reset_drag_tracking)

root.mainloop()