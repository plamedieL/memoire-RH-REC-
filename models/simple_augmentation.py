import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import math

class SimpleFaceAugmentation:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        
    def analyze_dataset(self):
        """Analyze the dataset to count images per person"""
        person_counts = {}
        total_persons = 0
        total_images = 0
        
        print("Analyzing dataset...")
        
        for person_folder in self.dataset_path.iterdir():
            if person_folder.is_dir():
                image_files = [f for f in person_folder.iterdir() 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                count = len(image_files)
                person_counts[person_folder.name] = count
                total_persons += 1
                total_images += count
                
        return person_counts, total_persons, total_images
    
    def get_needs_augmentation(self, person_counts, min_images=10):
        """Get list of persons who need augmentation"""
        needs_aug = []
        for person, count in person_counts.items():
            if count < min_images:
                needs_aug.append({
                    'person': person,
                    'current_count': count,
                    'needed': min_images - count
                })
        return sorted(needs_aug, key=lambda x: x['needed'], reverse=True)
    
    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        return image.rotate(angle, expand=True, fillcolor='white')
    
    def flip_image(self, image):
        """Flip image horizontally"""
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    def adjust_brightness(self, image, factor):
        """Adjust image brightness"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, image, factor):
        """Adjust image contrast"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def adjust_color(self, image, factor):
        """Adjust image color saturation"""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def add_noise(self, image):
        """Add simple noise to image"""
        import random
        pixels = list(image.getdata())
        noisy_pixels = []
        
        for pixel in pixels:
            if len(pixel) == 3:  # RGB
                r, g, b = pixel
                noise = random.randint(-20, 20)
                noisy_pixels.append((
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise))
                ))
            else:  # RGBA or other
                noisy_pixels.append(pixel)
        
        new_image = Image.new(image.mode, image.size)
        new_image.putdata(noisy_pixels)
        return new_image
    
    def blur_image(self, image):
        """Apply slight blur to image"""
        return image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    def crop_and_resize(self, image, crop_ratio=0.9):
        """Crop center and resize back to original"""
        width, height = image.size
        new_width = int(width * crop_ratio)
        new_height = int(height * crop_ratio)
        
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height), Image.Resampling.LANCZOS)
    
    def augment_single_image(self, image):
        """Apply random augmentations to a single image"""
        augmentation_functions = [
            lambda: self.rotate_image(image, random.uniform(-15, 15)),
            lambda: self.flip_image(image),
            lambda: self.adjust_brightness(image, random.uniform(0.8, 1.2)),
            lambda: self.adjust_contrast(image, random.uniform(0.8, 1.2)),
            lambda: self.adjust_color(image, random.uniform(0.8, 1.2)),
            lambda: self.add_noise(image),
            lambda: self.blur_image(image),
            lambda: self.crop_and_resize(image, random.uniform(0.85, 0.95)),
        ]
        
        # Apply 1-3 random augmentations
        num_augmentations = random.randint(1, 3)
        result = image
        
        for _ in range(num_augmentations):
            aug_func = random.choice(augmentation_functions)
            result = aug_func()
        
        return result
    
    def augment_person_images(self, person_name, target_count):
        """Augment images for a specific person to reach target count"""
        person_folder = self.dataset_path / person_name
        
        if not person_folder.exists():
            print(f"Folder not found: {person_name}")
            return False
            
        # Get existing images
        existing_images = [f for f in person_folder.iterdir() 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        current_count = len(existing_images)
        needed = target_count - current_count
        
        if needed <= 0:
            return True
            
        print(f"Augmenting {person_name}: {current_count} -> {target_count} (need {needed} more)")
        
        # Load existing images
        images = []
        for img_path in existing_images:
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if not images:
            print(f"No valid images found for {person_name}")
            return False
            
        # Generate augmented images
        generated_count = 0
        
        for i in range(needed):
            # Randomly select a source image
            source_img = random.choice(images)
            
            # Apply augmentation
            aug_img = self.augment_single_image(source_img)
            
            # Generate new filename
            new_number = current_count + i + 1
            new_filename = f"{person_name}_{new_number:04d}.jpg"
            new_filepath = person_folder / new_filename
            
            # Save augmented image
            try:
                aug_img.save(new_filepath, 'JPEG', quality=90)
                generated_count += 1
            except Exception as e:
                print(f"Error saving {new_filepath}: {e}")
            
        print(f"Generated {generated_count} augmented images for {person_name}")
        return generated_count == needed
    
    def process_dataset(self, min_images=10):
        """Process the entire dataset to ensure minimum images per person"""
        # Analyze dataset
        person_counts, total_persons, total_images = self.analyze_dataset()
        
        print(f"\nDataset Analysis:")
        print(f"Total persons: {total_persons}")
        print(f"Total images: {total_images}")
        print(f"Average images per person: {total_images/total_persons:.2f}")
        
        # Find persons needing augmentation
        needs_aug = self.get_needs_augmentation(person_counts, min_images)
        
        print(f"\nPersons needing augmentation (< {min_images} images): {len(needs_aug)}")
        
        if not needs_aug:
            print("All persons already have sufficient images!")
            return person_counts
        
        # Show top 10 persons needing most augmentation
        print("\nTop 10 persons needing most augmentation:")
        for i, person_info in enumerate(needs_aug[:10]):
            print(f"{i+1:2d}. {person_info['person']}: {person_info['current_count']} -> {min_images} (need {person_info['needed']})")
        
        # Process each person
        success_count = 0
        failed_count = 0
        
        print(f"\nStarting augmentation process...")
        
        for i, person_info in enumerate(needs_aug):
            person_name = person_info['person']
            target_count = min_images
            
            print(f"Progress: {i+1}/{len(needs_aug)} - Processing {person_name}")
            
            if self.augment_person_images(person_name, target_count):
                success_count += 1
                person_counts[person_name] = target_count
            else:
                failed_count += 1
                print(f"Failed to augment {person_name}")
        
        # Final analysis
        print(f"\nAugmentation Complete!")
        print(f"Successfully augmented: {success_count} persons")
        print(f"Failed: {failed_count} persons")
        
        # Recalculate totals
        new_total_images = sum(person_counts.values())
        added_images = new_total_images - total_images
        
        print(f"Original total images: {total_images}")
        print(f"New total images: {new_total_images}")
        print(f"Images added: {added_images}")
        
        return person_counts
    
    def verify_results(self, min_images=10):
        """Verify that all persons have at least the minimum number of images"""
        person_counts, _, total_images = self.analyze_dataset()
        
        below_min = []
        for person, count in person_counts.items():
            if count < min_images:
                below_min.append((person, count))
        
        print(f"\nVerification Results:")
        print(f"Total persons: {len(person_counts)}")
        print(f"Total images: {total_images}")
        print(f"Persons with < {min_images} images: {len(below_min)}")
        
        if below_min:
            print("\nPersons still below minimum:")
            for person, count in below_min:
                print(f"  {person}: {count}")
        else:
            print("✓ All persons have sufficient images!")
        
        return len(below_min) == 0

def main():
    # Set the dataset path
    dataset_path = "archive/lfw-deepfunneled/lfw-deepfunneled"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
    
    # Create augmentation instance
    augmenter = SimpleFaceAugmentation(dataset_path)
    
    # Process the dataset
    print("Starting face data augmentation process...")
    print("=" * 50)
    
    try:
        # Process with minimum 10 images per person
        final_counts = augmenter.process_dataset(min_images=10)
        
        print("\n" + "=" * 50)
        print("Verifying results...")
        
        # Verify the results
        success = augmenter.verify_results(min_images=10)
        
        if success:
            print("\n🎉 Data augmentation completed successfully!")
            print("All persons now have at least 10 images.")
        else:
            print("\n⚠️ Some persons may still need attention.")
            
    except Exception as e:
        print(f"Error during augmentation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
