import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
from datetime import datetime
import pandas as pd
import easyocr
import imutils

class HighAccuracyTextExtractor:
    def __init__(self):
        self.output_dir = "high_accuracy_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results_df = pd.DataFrame(columns=['Timestamp', 'Extracted_Text', 'Confidence', 'Image_Path'])
        
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False)  

    def advanced_preprocessing(self, image_path):
        """Advanced image preprocessing for maximum OCR accuracy"""
        img = cv2.imread(image_path)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        def deskew(image):
            angle = cv2.minAreaRect(np.column_stack(np.where(image > 0)))[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            rotated = imutils.rotate_bound(image, angle)
            return rotated
        
        # Apply multiple preprocessing steps
        # 1. Deskew
        gray = deskew(gray)
        
        # 2. Adaptive thresholding with larger block size
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        
        # 3. Morphological operations to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 4. Noise removal with bilateral filter
        filtered = cv2.bilateralFilter(morph, 9, 75, 75)
        
        # 5. Sharpen image
        sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(filtered, -1, sharpening_kernel)
        
        return sharpened, img

    def extract_text_combined(self, image_path):
        """Combine multiple OCR methods for better accuracy"""
        try:
            # Preprocess image
            processed_img, original_img = self.advanced_preprocessing(image_path)
            
            # Method 1: Tesseract with multiple configurations
            tesseract_configs = [
                r'--oem 3 --psm 6',  # Default text block
                r'--oem 3 --psm 3',  # Single uniform block
                r'--oem 3 --psm 11'  # Sparse text
            ]
            
            tesseract_results = []
            for config in tesseract_configs:
                text = pytesseract.image_to_string(processed_img, config=config, lang='eng')
                if text.strip():
                    tesseract_results.append(text.strip())
            
            # Method 2: EasyOCR
            easy_results = self.easyocr_reader.readtext(processed_img, detail=1)
            easy_text = " ".join([result[1] for result in easy_results])
            easy_confidence = np.mean([result[2] for result in easy_results]) if easy_results else 0
            
            # Combine results - prefer EasyOCR if confidence is high, otherwise use longest Tesseract result
            final_text = easy_text if easy_confidence > 0.7 and easy_text.strip() else \
                        max(tesseract_results, key=len, default="") if tesseract_results else ""
            
            return final_text, max(easy_confidence, 0.7), processed_img, original_img
            
        except Exception as e:
            return f"Error: {str(e)}", 0, None, None

    def display_and_store(self, image_path):
        """Display and store results with confidence"""
        text, confidence, processed_img, original_img = self.extract_text_combined(image_path)
        
        if processed_img is not None and original_img is not None:
            # Display images with overlay text
            cv2.putText(original_img, f"Original (Conf: {confidence:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Split text into lines for display
            lines = text.split('\n')
            for i, line in enumerate(lines[:5]):  # Show first 5 lines
                cv2.putText(processed_img, line[:50],  # Limit line length
                           (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Show images
            cv2.imshow("Original", original_img)
            cv2.imshow("Processed with Text", processed_img)
            
            print("\nExtracted Text (Confidence: {:.2f}):".format(confidence))
            print("-" * 60)
            print(text)
            print("-" * 60)
            
            # Store results
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            result_dict = {
                'Timestamp': timestamp,
                'Extracted_Text': text,
                'Confidence': confidence,
                'Image_Path': image_path
            }
            
            self.results_df = pd.concat([self.results_df, pd.DataFrame([result_dict])], 
                                      ignore_index=True)
            
            csv_path = os.path.join(self.output_dir, "high_accuracy_results.csv")
            self.results_df.to_csv(csv_path, index=False)
            
            img_save_path = os.path.join(self.output_dir, f"processed_{timestamp}.jpg")
            cv2.imwrite(img_save_path, processed_img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return text, confidence

def main():
    extractor = HighAccuracyTextExtractor()
    image_path = "WhatsApp Image 2025-03-21 at 14.27.08_131a5436.jpg"  # Replace with your image path
    
    if os.path.exists(image_path):
        print(f"Processing image: {image_path}")
        text, confidence = extractor.display_and_store(image_path)
        print(f"Final result - Confidence: {confidence:.2f}")
    else:
        print("Image file not found! Please provide a valid image path.")

if __name__ == "__main__":
    main()