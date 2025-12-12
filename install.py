"""
Installation script for Enhanced AR Vision System
"""
import subprocess
import sys
import os
import platform
import time
from pathlib import Path

class Installer:
    def __init__(self):
        self.system = platform.system()
        self.is_windows = self.system == "Windows"
        self.is_linux = self.system == "Linux"
        self.is_macos = self.system == "Darwin"
        
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        
        # Colors for console output
        self.GREEN = "\033[92m"
        self.YELLOW = "\033[93m"
        self.RED = "\033[91m"
        self.BLUE = "\033[94m"
        self.RESET = "\033[0m"
        self.BOLD = "\033[1m"
    
    def print_header(self, text):
        """Print colored header"""
        print(f"\n{self.BOLD}{self.BLUE}{'='*70}{self.RESET}")
        print(f"{self.BOLD}{self.BLUE}{text}{self.RESET}")
        print(f"{self.BOLD}{self.BLUE}{'='*70}{self.RESET}\n")
    
    def print_success(self, text):
        """Print success message"""
        print(f"{self.GREEN}✓ {text}{self.RESET}")
    
    def print_warning(self, text):
        """Print warning message"""
        print(f"{self.YELLOW}⚠ {text}{self.RESET}")
    
    def print_error(self, text):
        """Print error message"""
        print(f"{self.RED}✗ {text}{self.RESET}")
    
    def print_info(self, text):
        """Print info message"""
        print(f"{self.BLUE}ℹ {text}{self.RESET}")
    
    def check_python_version(self):
        """Check Python version"""
        self.print_info("Checking Python version...")
        
        if sys.version_info < (3, 8):
            self.print_error("Python 3.8 or higher is required")
            self.print_info(f"Current version: {sys.version}")
            return False
        
        self.print_success(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True
    
    def check_system_requirements(self):
        """Check system requirements"""
        self.print_info("Checking system requirements...")
        
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.print_success(f"GPU detected: {torch.cuda.get_device_name(0)}")
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.print_success("Apple MPS (Metal) detected")
                return "mps"
            else:
                self.print_warning("No GPU detected, using CPU (performance will be limited)")
                return "cpu"
        except:
            self.print_warning("Could not check GPU, assuming CPU")
            return "cpu"
    
    def create_requirements_file(self):
        """Create requirements.txt file"""
        requirements = """# Enhanced AR Vision System Requirements
opencv-python>=4.8.0
mediapipe>=0.10.8
ultralytics>=8.0.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
matplotlib>=3.7.0
scipy>=1.11.0
tqdm>=4.65.0
"""
        
        try:
            with open(self.requirements_file, 'w') as f:
                f.write(requirements)
            self.print_success("Created requirements.txt")
            return True
        except Exception as e:
            self.print_error(f"Failed to create requirements.txt: {e}")
            return False
    
    def install_packages(self):
        """Install required packages"""
        self.print_header("Installing Python Packages")
        
        # Create requirements.txt if it doesn't exist
        if not self.requirements_file.exists():
            if not self.create_requirements_file():
                return False
        
        # Upgrade pip first
        self.print_info("Upgrading pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            self.print_success("pip upgraded")
        except subprocess.CalledProcessError:
            self.print_warning("Could not upgrade pip, continuing...")
        
        # Install requirements
        self.print_info("Installing packages from requirements.txt...")
        
        packages = [
            ("opencv-python", ">=4.8.0"),
            ("mediapipe", ">=0.10.8"),
            ("ultralytics", ">=8.0.0"),
            ("torch", ">=2.0.0"),
            ("torchvision", ">=0.15.0"),
            ("numpy", ">=1.24.0"),
            ("Pillow", ">=10.0.0"),
        ]
        
        for package, version in packages:
            try:
                self.print_info(f"Installing {package}...")
                if self.is_windows:
                    # Windows specific installation
                    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}{version}"])
                else:
                    # Linux/Mac
                    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}{version}"])
                self.print_success(f"Installed: {package}")
            except subprocess.CalledProcessError as e:
                self.print_error(f"Failed to install {package}: {e}")
                
                # Try without version specifier
                try:
                    self.print_info(f"Trying to install {package} without version constraint...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    self.print_success(f"Installed: {package} (without version constraint)")
                except:
                    self.print_error(f"Failed to install {package} completely")
                    return False
        
        return True
    
    def install_torch_with_cuda(self):
        """Install PyTorch with CUDA support if available"""
        self.print_info("Checking CUDA availability for PyTorch...")
        
        # Check if CUDA is already installed via nvidia-smi
        if self.is_windows or self.is_linux:
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    self.print_success("NVIDIA GPU detected")
                    
                    # Get CUDA version
                    try:
                        cuda_version_output = subprocess.run(['nvcc', '--version'], 
                                                           capture_output=True, text=True)
                        if 'release' in cuda_version_output.stdout:
                            cuda_version = cuda_version_output.stdout.split('release ')[1].split(',')[0]
                            self.print_info(f"CUDA version: {cuda_version}")
                            
                            # Install appropriate PyTorch version
                            cuda_major = cuda_version.split('.')[0]
                            if cuda_major == '11':
                                torch_command = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
                            elif cuda_major == '12':
                                torch_command = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
                            else:
                                torch_command = "torch torchvision torchaudio"
                            
                            self.print_info("Installing PyTorch with CUDA support...")
                            subprocess.check_call([sys.executable, "-m", "pip", "install"] + torch_command.split())
                            self.print_success("PyTorch with CUDA installed")
                            return True
                    except:
                        pass
            except FileNotFoundError:
                self.print_warning("nvidia-smi not found, assuming no GPU")
        
        return False
    
    def check_yolo_model(self):
        """Check and download YOLO model"""
        self.print_header("Checking YOLO Models")
        
        model_files = [
            "yolov8x.pt",  # Best accuracy
            "yolov8l.pt",  # Good balance
            "yolov8m.pt",  # Medium
            "yolov8s.pt",  # Small
            "yolov8n.pt",  # Nano (fastest)
        ]
        
        existing_models = []
        for model in model_files:
            model_path = self.project_root / model
            if model_path.exists():
                existing_models.append(model)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                self.print_success(f"{model} exists ({size_mb:.1f} MB)")
        
        if existing_models:
            self.print_info(f"Found {len(existing_models)} YOLO model(s)")
            return True
        
        # Download a model
        self.print_info("No YOLO models found. Downloading...")
        
        models_to_download = ["yolov8n.pt", "yolov8s.pt"]  # Download smaller models first
        
        for model in models_to_download:
            try:
                self.print_info(f"Downloading {model}...")
                
                # Using ultralytics to download
                import ultralytics
                from ultralytics import YOLO
                
                # Download model
                yolo = YOLO(model)
                
                # Save it locally
                model_path = self.project_root / model
                if hasattr(yolo.model, 'save'):
                    yolo.model.save(model_path)
                
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                self.print_success(f"Downloaded {model} ({size_mb:.1f} MB)")
                return True
                
            except Exception as e:
                self.print_error(f"Failed to download {model}: {e}")
        
        # Manual download instructions
        self.print_warning("""
Could not automatically download YOLO models.
Please download manually:
1. Go to: https://github.com/ultralytics/ultralytics
2. Download one of these models:
   - yolov8n.pt (smallest, fastest)
   - yolov8s.pt (good balance)
   - yolov8x.pt (largest, most accurate)
3. Place the .pt file in this folder
""")
        
        return False
    
    def create_project_structure(self):
        """Create necessary project folders"""
        self.print_info("Creating project structure...")
        
        folders = ["models", "ar", "utils", "data", "screenshots"]
        
        for folder in folders:
            folder_path = self.project_root / folder
            try:
                folder_path.mkdir(exist_ok=True)
                
                # Create __init__.py for Python packages
                if folder in ["models", "ar", "utils"]:
                    init_file = folder_path / "__init__.py"
                    init_file.touch(exist_ok=True)
                
                self.print_success(f"Created folder: {folder}")
            except Exception as e:
                self.print_error(f"Failed to create {folder}: {e}")
        
        # Create example files if they don't exist
        example_files = {
            "main.py": """# Enhanced AR Vision System
from ar_vision_system import ARVisionSystem

if __name__ == "__main__":
    app = ARVisionSystem()
    app.run()
""",
            "config.py": """# Configuration file
CONFIG = {
    'yolo_model': 'yolov8n.pt',
    'confidence_threshold': 0.3,
}
""",
            "README.md": """# Enhanced AR Vision System

Real-time object detection with AR capabilities.

## Features
- Real-time object detection with YOLOv8
- Hand gesture recognition
- Human activity detection
- AR object manipulation

## Installation
python install.py

## Usage
python main.py
"""
        }
        
        for filename, content in example_files.items():
            file_path = self.project_root / filename
            if not file_path.exists():
                try:
                    with open(file_path, 'w') as f:
                        f.write(content)
                    self.print_success(f"Created: {filename}")
                except Exception as e:
                    self.print_error(f"Failed to create {filename}: {e}")
    
    def test_installation(self):
        """Test if installation was successful"""
        self.print_header("Testing Installation")
        
        tests = [
            ("OpenCV", "cv2"),
            ("NumPy", "numpy"),
            ("PyTorch", "torch"),
            ("Ultralytics", "ultralytics"),
            ("MediaPipe", "mediapipe"),
        ]
        
        all_passed = True
        
        for name, module in tests:
            try:
                __import__(module)
                self.print_success(f"{name}: OK")
            except ImportError as e:
                self.print_error(f"{name}: FAILED - {e}")
                all_passed = False
        
        # Test YOLO model loading
        try:
            from ultralytics import YOLO
            model_files = list(self.project_root.glob("*.pt"))
            if model_files:
                model = YOLO(str(model_files[0]))
                self.print_success("YOLO model: OK")
            else:
                self.print_warning("YOLO model: No .pt files found")
        except Exception as e:
            self.print_error(f"YOLO model: FAILED - {e}")
            all_passed = False
        
        return all_passed
    
    def setup_environment(self):
        """Setup environment variables if needed"""
        self.print_info("Setting up environment...")
        
        # Add current directory to Python path
        python_path = os.environ.get('PYTHONPATH', '')
        current_dir = str(self.project_root)
        
        if current_dir not in python_path:
            os.environ['PYTHONPATH'] = f"{current_dir}{os.pathsep}{python_path}"
            self.print_info(f"Added {current_dir} to PYTHONPATH")
        
        # Create .env file for configuration
        env_file = self.project_root / ".env"
        if not env_file.exists():
            try:
                with open(env_file, 'w') as f:
                    f.write("# AR Vision System Environment Variables\n")
                    f.write("USE_GPU=True\n")
                    f.write("MODEL_PATH=yolov8n.pt\n")
                    f.write("CONFIDENCE_THRESHOLD=0.3\n")
                self.print_success("Created .env file")
            except Exception as e:
                self.print_error(f"Failed to create .env: {e}")
    
    def print_final_instructions(self):
        """Print final instructions"""
        self.print_header("Installation Complete!")
        
        print(f"""
{self.BOLD}{self.GREEN}🎉 Enhanced AR Vision System is ready!{self.RESET}

{self.BOLD}Next Steps:{self.RESET}
1. {self.YELLOW}Connect a webcam{self.RESET} (if not already connected)
2. Run the application:
   {self.BLUE}python main.py{self.RESET}

{self.BOLD}Controls:{self.RESET}
  Q - Quit application
  R - Reset interactions
  +/- - Adjust confidence
  F - Toggle fullscreen
  SPACE - Take screenshot
  T - Toggle object tracking

{self.BOLD}For better performance:{self.RESET}
• Ensure good lighting
• Objects should be clearly visible
• Use yolov8n.pt for speed, yolov8x.pt for accuracy
• Adjust confidence threshold with +/- keys

{self.BOLD}Need help?{self.RESET}
• Check the README.md file
• Make sure your webcam is properly connected
• Try different YOLO models for speed/accuracy balance

{self.BOLD}Troubleshooting:{self.RESET}
• If detection is slow, try yolov8n.pt
• If objects aren't detected, increase confidence threshold (+ key)
• If webcam isn't working, check device permissions
""")
    
    def run(self):
        """Run the complete installation process"""
        self.print_header("Enhanced AR Vision System - Installation")
        
        print(f"{self.BOLD}System: {self.system}{self.RESET}")
        print(f"{self.BOLD}Python: {sys.version}{self.RESET}")
        print(f"{self.BOLD}Project Directory: {self.project_root}{self.RESET}")
        
        # Run installation steps
        steps = [
            ("Python Version Check", self.check_python_version),
            ("System Requirements", self.check_system_requirements),
            ("Project Structure", self.create_project_structure),
            ("Installing Packages", self.install_packages),
            ("YOLO Models", self.check_yolo_model),
            ("Environment Setup", self.setup_environment),
            ("Testing Installation", self.test_installation),
        ]
        
        for step_name, step_func in steps:
            self.print_info(f"Step: {step_name}")
            if not step_func():
                self.print_error(f"Installation failed at: {step_name}")
                if step_name != "Testing Installation":  # Test might fail but continue
                    print(f"\n{self.RED}Installation failed. Please fix the errors above.{self.RESET}")
                    return False
        
        self.print_final_instructions()
        return True

def main():
    """Main entry point"""
    installer = Installer()
    
    try:
        success = installer.run()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()