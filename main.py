"""
Qwen2.5-Coder Evolution System - Main Entry Point

Project Structure:
├── src/
│   ├── config/           # Configuration module
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── utils/            # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py     # Logging system
│   │   ├── api_helper.py # API helper
│   │   ├── code_tools.py # Code processing tools
│   │   └── qa_interface.py # Q&A interface
│   ├── models/           # Model loading
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── training/         # Training module
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/       # Evaluation module
│   │   ├── __init__.py
│   │   └── evaluator.py
│   └── ui/              # UI Interface
│       ├── __init__.py
│       └── interface.py
├── datasets/            # Dataset directory
├── models/              # Model directory
└── main.py             # Main program
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check required dependencies"""
    required_packages = ["torch", "transformers", "gradio"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("[ERROR] Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install with:")
        print("pip install torch transformers gradio")
        print("\nOptional packages (for complete features):")
        print("pip install datasets accelerate peft requests")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    from src.config.settings import DEFAULT_CONFIG
    
    directories = [
        "./models",
        "./datasets",
        "./mbpp_training_data",
        DEFAULT_CONFIG["output_dir"]
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Directory created: {directory}")

def check_datasets():
    """Check datasets"""
    from src.config.settings import DEFAULT_CONFIG
    
    mbpp_path = DEFAULT_CONFIG["mbpp_dataset_path"]
    
    # Check MBPP dataset
    if not os.path.exists(mbpp_path):
        print(f"[WARNING] MBPP dataset not found: {mbpp_path}")
        print("[INFO] Creating new MBPP dataset file")
        
        # Create directory
        os.makedirs(os.path.dirname(mbpp_path), exist_ok=True)
        
        # Create sample MBPP dataset
        with open(mbpp_path, 'w', encoding='utf-8') as f:
            f.write('"Write a function to add two numbers and return the sum"\n')
            f.write('"Write a function to check if a number is prime"\n')
            f.write('"Write a function to generate the first n Fibonacci numbers"\n')
        print(f"[OK] Sample MBPP dataset created: {mbpp_path}")
    else:
        print(f"[OK] MBPP dataset found: {mbpp_path}")
    
    # Check HumanEval dataset
    human_eval_path = DEFAULT_CONFIG["human_eval_path"]
    if not os.path.exists(human_eval_path):
        print(f"[WARNING] HumanEval dataset not found: {human_eval_path}")
        print("[INFO] Download from:")
        print("https://github.com/openai/human-eval")
        print("[INFO] Save to ./datasets/ directory")
    else:
        print(f"[OK] HumanEval dataset found: {human_eval_path}")

def main():
    """Main program entry point"""
    print("="*60)
    print("[SYSTEM] Qwen2.5-Coder Evolution System")
    print("="*60)
    
    # Check dependencies
    print("\n[1/3] Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("[OK] All dependencies installed\n")
    
    # Create directories
    print("[2/3] Creating project directories...")
    create_directories()
    
    # Check datasets
    print("\n[3/3] Checking datasets...")
    check_datasets()
    
    # Launch interface
    print("\n" + "="*60)
    print("[INFO] Starting Qwen2.5-Coder Evolution System...")
    print("="*60)
    print("\n[INFO] Access at: http://localhost:7860")
    print("[INFO] Press Ctrl+C to stop server\n")
    
    try:
        from src.ui import create_interface
        
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_api=False
        )
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped")
    except Exception as e:
        print(f"\n[ERROR] Failed to start: {str(e)}")
        logger.exception("Error launching interface")
        sys.exit(1)

if __name__ == "__main__":
    main()
