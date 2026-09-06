import numpy as np

# Configure print options
np.set_printoptions(suppress=True, precision=4)

data = np.load("MatrixValues.npz")
arrayName = input("Array Key: ")

if arrayName in data:
    arr = data[arrayName]
    
    print(f"\n--- {arrayName} Details ---")
    print(f"Shape: {arr.shape}")
    print(f"Total Elements: {arr.size}")
    print(f"Memory Size: {arr.nbytes:,} bytes")
    
    # Safety threshold to prevent terminal freezing on massive arrays
    MAX_ELEMENTS = 5000
    
    if arr.size > MAX_ELEMENTS:
        print(f"\n[Notice] Array is too large ({arr.size:,} elements) for a full printout.")
        print("Displaying a preview (first 5 rows) instead:\n")
        print(arr[:5])
    else:
        print("\nFull Array Data:\n")
        print(arr)
else:
    print(f"Key '{arrayName}' not found in the file.")