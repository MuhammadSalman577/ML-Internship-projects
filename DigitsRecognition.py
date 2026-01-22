import os 
import cv2 
import tensorflow as neuralNet 
import numpy as np 
import matplotlib.pyplot as plt 

model = neuralNet.keras.models.load_model("handwritten.keras") 
ImagesList = ['One','Two','Three','Four','Five','Six','Seven','Eight','Nine','anotherNine','anotherEight','anotherSeven'] 

def preprocess_image(image_path):
    """Preprocess image to match MNIST format with enhanced cleaning"""
    # Read grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Invert if needed (digit should be white on black)
    if np.mean(img) > 127:
        img = 255 - img
    
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Binary threshold with OTSU
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Find bounding box of the digit
    coords = cv2.findNonZero(img)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]
    
    # Add some padding before resizing (helps with edge cases)
    pad = 5
    cropped = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    
    # Make it square by padding the shorter dimension
    h, w = cropped.shape
    if h > w:
        diff = h - w
        cropped = cv2.copyMakeBorder(cropped, 0, 0, diff//2, diff - diff//2, cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        diff = w - h
        cropped = cv2.copyMakeBorder(cropped, diff//2, diff - diff//2, 0, 0, cv2.BORDER_CONSTANT, value=0)
    
    # Resize to 20x20
    resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)
    
    # Create 28x28 black canvas and center the digit
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = resized
    
    # Normalize to 0-1
    canvas = canvas.astype('float32') / 255.0
    
    return canvas

correct = 0
total = 0

for name in ImagesList: 
    path = f"HandWrittenDigits/{name}.png" 
    if os.path.isfile(path): 
        try: 
            img = preprocess_image(path)
            
            if img is None:
                print(f"{name} -> No digit found in image")
                continue
            
            total += 1
            
            # Reshape for model input
            img_input = img.reshape(1, 28, 28, 1)
            
            # Predict
            pred = model.predict(img_input, verbose=0) 
            predicted_digit = np.argmax(pred)
            confidence = np.max(pred) * 100
            
            # Show top 3 predictions
            top3_idx = np.argsort(pred[0])[-3:][::-1]
            top3_probs = [(i, pred[0][i]*100) for i in top3_idx]
            
            # Try to extract expected digit from filename
            digit_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5, 
                        'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9, 
                        'anotherNine': 9, 'anotherEight': 8}
            
            expected = digit_map.get(name, None)
            is_correct = "✓" if expected == predicted_digit else "✗"
            
            if expected == predicted_digit:
                correct += 1
            
            print(f"\n{name} -> Predicted: {predicted_digit} ({confidence:.2f}%) {is_correct}")
            if expected is not None:
                print(f"  Expected: {expected}")
            print(f"  Top 3: {top3_probs[0][0]}:{top3_probs[0][1]:.1f}%, {top3_probs[1][0]}:{top3_probs[1][1]:.1f}%, {top3_probs[2][0]}:{top3_probs[2][1]:.1f}%")
            
            # Display
            plt.imshow(img, cmap='gray') 
            plt.title(f"{name} -> Predicted: {predicted_digit} ({confidence:.1f}%) {is_correct}") 
            plt.axis('off')
            plt.show() 
            
        except Exception as e: 
            print(f"Error with {name}:", e) 
    else: 
        print("Path not found:", path)

print(f"\n{'='*50}")
print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
print(f"{'='*50}")