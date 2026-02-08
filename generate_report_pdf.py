from fpdf import FPDF
import os

class DeepSeaReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, 'DeepSea Restoration AI: Technical Project Compendium', 0, 0, 'R')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

def generate_10_page_report():
    pdf = DeepSeaReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    
    # helper for adding Q&A sections
    def add_qa(q, a):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 119, 190)
        pdf.multi_cell(170, 7, text=f"Question: {q}")
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(170, 7, text=f"Answer: {a}")
        pdf.ln(6)

    # helper for adding section titles
    def add_section_title(text):
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(0, 70, 120)
        pdf.multi_cell(170, 12, text=text, align='L')
        pdf.ln(5)

    # helper for adding subheadings
    def add_subheading(text):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(170, 8, text=text, align='L')
        pdf.ln(2)

    # helper for adding text
    def add_text(text):
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(50, 50, 50)
        text = text.replace("—", "--").replace("“", '"').replace("”", '"').replace("–", "-")
        pdf.multi_cell(170, 7, text=text)
        pdf.ln(5)

    # ---------------------------------------------------------
    # PAGE 1: COVER PAGE
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 119, 190)
    pdf.multi_cell(170, 15, text="DEEPSEA RESTORATION AI", align='C')
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(170, 10, text="A Hybrid Variational-Neural Framework for High-Fidelity Underwater Image Enhancement", align='C')
    pdf.ln(40)
    pdf.set_font("Helvetica", "I", 12)
    pdf.multi_cell(170, 10, text="Technical Compendium and Project Documentation", align='C')
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(170, 8, text="Developed for Advanced Marine Imaging Research\nProject Version: 2.1.0\nOperating Environment: PyTorch / OpenCV / Gradio", align='C')
    
    # ---------------------------------------------------------
    # PAGE 2: ABSTRACT & INTRODUCTION
    # ---------------------------------------------------------
    pdf.add_page()
    add_section_title("1. ABSTRACT")
    add_text("Underwater optical imaging is significantly hampered by the physical properties of the aquatic medium. Light traveling through water undergoes frequency-dependent absorption and scattering, resulting in three primary degradations: severe color casting (typically blue or green), low global contrast, and a 'haze' effect known as turbidity. Traditional image processing methods often fail to recover true colors, while standard deep learning models frequently introduce artifacts or hallucinate details. This project proposes a hybrid framework that bridges the gap between physics-based variational imaging models and data-driven neural refinement.\n\nThe core contribution of this research is the implementation of a dual-stage restoration engine. By first applying an underwater light attenuation model to perform a coarse restoration, the system removes the majority of the liquid medium's interference. Then, using a specialized Residual U-Net for neural polishing, we achieve unprecedented results in underwater visibility, color fidelity, and structural preservation. This document serves as the complete technical manual for the system.")
    
    add_section_title("2. INTRODUCTION")
    add_text("Over 70% of our planet is covered by water, yet our ability to visually explore the ocean floor is limited by the very medium we wish to investigate. Marine scientists, ROV pilots, and ocean researchers rely on visual data to monitor coral reef health, identify shipwrecks, and study biological species. However, raw underwater footage is almost always unusable without heavy post-processing.\n\nHistorically, photographers used color filters or expensive strobe lighting to compensate for light loss. In the digital age, we have early algorithms like 'Global Histogram Equalization' or 'Retinex,' but these often explode noise in the dark areas. This project moves beyond simple filters into 'Hybrid Intelligence' -- using the laws of physics to undo what the water did, and the power of AI to fix what the math missed. The proposed system is designed to be hardware-aware and real-time capable.")

    # ---------------------------------------------------------
    # PAGE 3: UNDERWATER OPTICS & CHALLENGES
    # ---------------------------------------------------------
    pdf.add_page()
    add_section_title("3. UNDERWATER OPTICS & CHALLENGES")
    add_subheading("3.1 Frequency-Dependent Absorption")
    add_text("Water molecules and dissolved organic matter absorb light photons as thermal energy. Critically, this absorption is not uniform across the visible spectrum. Red light (longer wavelengths) has a very high absorption coefficient and is absorbed almost entirely within the first 5 meters of depth. Orange and yellow follow shortly after. This is why most underwater images appear overwhelmingly cyan, blue, or green. Our project implements a system that specifically targets the recovery of these 'lost' wavelengths by calculating the absorption differential.")
    
    add_subheading("3.2 Backscattering and Turbidity")
    add_text("Backscattering occurs when light hits floating particles (marine snow, sediments, algae) and reflects directly back into the camera lens before ever reaching the object. This creates a 'veiling glare' or haze that reduces contrast to near-zero as the distance between the camera and the object increases. This effect is mathematically similar to fog on land but is much more severe due to the density of the medium. Our Variational Model uses the Dark Channel Prior (DCP) principles to quantify this backscatter and subtract it from the scene effectively.")
    
    add_subheading("3.3 Color Distortion")
    add_text("Beyond simple absorption, the refraction index of salt water differs from air, causing slight distortions in geometric clarity. While our primary goal is color and visibility, the SSIM (Structural Similarity Index) component of our loss function ensures that the geometric shapes of corals and marine life are preserved with high fidelity throughout the restoration process.")

    # ---------------------------------------------------------
    # PAGE 4: THE PHYSICAL MODEL (STAGE 1)
    # ---------------------------------------------------------
    pdf.add_page()
    add_section_title("4. STAGE 1: VARIATIONAL PHYSICS MODEL")
    add_text("The first stage of the project is the 'Physical Dehazing Engine.' It relies on the Atmospheric Scattering Model, adapted for underwater environments. Unlike AI, which 'guesses,' this stage uses deterministic mathematics to revert light patterns based on the Optical Channel Prior.")
    
    add_subheading("4.1 Backlight Estimation (A)")
    add_text("To remove the color cast, we must first find what color the 'Cast' is. The code identifies the Background Light (A) by searching for the most turbid areas of the image. It uses a selective search algorithm that picks the top 0.1% brightest pixels in the dark channel. This ensures that bright foreground objects are not mistaken for the background water color. This value (A) serves as the baseline for color subtraction across all three RGB channels.")
    
    add_subheading("4.2 Transmission Heatmap (t)")
    add_text("Modern physics tells us that depth equals degradation. The Transmission Map acts as a proxy for depth. Areas that are far away have low transmission, meaning the physical signal is mostly 'lost' to the water. We generate a heatmap where blue indicates deep, hazy water and red indicates clear objects close to the camera. This map is the key input for our restoration formula, allowing us to 'boost' the signal only where it is needed, preventing 'over-exposure' of close objects.")

    # ---------------------------------------------------------
    # PAGE 5: DEEP LEARNING ARCHITECTURE (STAGE 2)
    # ---------------------------------------------------------
    pdf.add_page()
    add_section_title("5. STAGE 2: NEURAL RESIDUAL REFINEMENT")
    add_subheading("5.1 The Residual U-Net Architecture")
    add_text("At the heart of our AI is a 23-layer Residual U-Net. This architecture is unique because it features a symmetric 'U' shape of Encoder and Decoder blocks. The Encoder extracts high-level 'semantic' information (what the object is), while the Decoder reconstructs the high-fidelity image. Crucially, we use 'Skip Connections' that pass raw spatial data across the U-bridge, ensuring that tiny details like coral textures or fish scales are never lost during processing.")
    
    add_subheading("5.2 Learning the Residual")
    add_text("Most AIs try to turn a blurry image into a clear one from scratch. Our project uses 'Residual Learning.' The AI only tries to find the MISTAKES made by the physics model in Stage 1. It predicts a 'Residual Map' (pixel corrections). By adding this map back to the physical result, we get an image that is scientifically accurate but visually stunning. This reduces the 'workload' on the AI and leads to much faster convergence during the training phase.")
    
    add_subheading("5.3 Activation Functions")
    add_text("We utilize LeakyReLU activation functions throughout the hidden layers to prevent 'Dead Neurons' and a Sigmoid activation at the final layer. The Sigmoid layer is critical as it bounds the final pixel values precisely between 0 and 1, matching the standard float32 representation of image tensors.")

    # ---------------------------------------------------------
    # PAGE 6: DATASET INTELLIGENCE & UIEB
    # ---------------------------------------------------------
    pdf.add_page()
    add_section_title("6. DATASET INTELLIGENCE: THE 'WHY' BEHIND UIEB")
    add_subheading("6.1 Why UIEB (The Gold Standard)?")
    add_text("Dataset selection is the single most important factor in AI reliability. Many researchers use 'Synthetic' datasets (where they take a surface picture and turn it blue in Photoshop). This is bad practice because Photoshop doesn't follow water physics. We use the UIEB (Underwater Image Enhancement Benchmark) dataset. It consist of real-world images taken by divers. Each raw image has a 'Reference' image that was hand-selected by marine biologists as the perfect restoration. This teaches our AI real-world light behavior and expert-level aesthetic quality.")
    
    add_subheading("6.2 Smart Data Controller & Cleaning")
    add_text("Real-world data is messy. Our project includes a 'Smart Data Controller' (src/dataset.py) that automatically handles:\n- Recursive Folder Scanning: It searches indefinitely through subfolders to find images.\n- Zombie File Filtering: It detects and ignores 0-byte or corrupted image files during the validation phase.\n- Robust Read Pathing: Uses np.fromfile to bypass Windows path length limits and special characters that usually crash standard OpenCV calls.")

    # ---------------------------------------------------------
    # PAGE 7: PROJECT EXECUTION GUIDE (HOW TO DO)
    # ---------------------------------------------------------
    pdf.add_page()
    add_section_title("7. HOW TO DO: PROJECT EXECUTION GUIDE")
    add_text("This section provides the exact technical steps to replicate, train, and run the project from scratch. It is designed for researchers who wish to deploy this in their own environments.")
    
    add_subheading("7.1 Environment Setup")
    add_text("1. Python Infrastructure: Install Python 3.9 through 3.12.\n2. Virtualization: Create a virtual environment using 'python -m venv .venv'.\n3. Dependency Installation: Execute 'pip install -r requirements.txt'. This installs PyTorch, OpenCV, Gradio, and Plotly.")
    
    add_subheading("7.2 Training the Model")
    add_text("To train the brain of the system, use: 'python train.py --epochs 50 --batch_size 4'. This script will scan the data directory, pair images using our 'Fuzzy Stem Matcher', and begin optimization. The state is saved every epoch, and the 'best_model.pth' is kept as the final checkpoint.")
    
    add_subheading("7.3 Launching the Dashboard")
    add_text("To use the project, run: 'python app.py'. This launches a local Gradio server. You can access it via your browser at http://127.0.0.1:7860. The dashboard is fully responsive and supports drag-and-drop image uploads.")

    # ---------------------------------------------------------
    # PAGE 8: PREMIUM DASHBOARD & ANALYTICS
    # ---------------------------------------------------------
    pdf.add_page()
    add_section_title("8. PREMIUM DASHBOARD & ANALYTICS")
    add_subheading("8.1 Performance Profiling")
    add_text("The dashboard (app.py) features real-time performance analytics using Plotly. It calculates the execution time for both the Variational and Neural stages. This allows developers to see where the bottlenecks are. It also predicts performance for higher resolutions, which is vital for planning real-time ROV missions.")
    
    add_subheading("8.2 Pipeline Inspection")
    add_text("Users can see the 'Guts' of the AI. You can view the input, the Heatmap, the coarse result, and the final output. This transparency is part of 'Explainable AI' (XAI), helping researchers understand exactly why a certain pixel was changed in a certain way.")
    
    add_subheading("8.3 Resolution Independence")
    add_text("Although the AI is trained on 256x256 tiles, the dashboard uses a 'Fully Convolutional' inference method. This means you can upload a 1080p or 4K image, and the system will process it without resizing or losing quality.")

    # ---------------------------------------------------------
    # PAGE 9: VIVA VOCE PREPARATION (FAQ)
    # ---------------------------------------------------------
    pdf.add_page()
    add_section_title("9. VIVA VOCE: FREQUENTLY ASKED QUESTIONS")
    
    add_qa("What is the primary motivation for this project?", "To enable marine researchers to capture high-clarity data for coral reef monitoring and biological surveys where natural lighting is insufficient and water turbidity is high.")
    add_qa("Why is a hybrid model better than a pure deep learning model?", "Pure deep learning is a 'black box' and often hallucinates details or artifacts. Adding a physics-based first stage ensures the AI stays grounded in the real laws of light.")
    add_qa("How does the model handle different water colors?", "The Stage 1 engine identifies the Atmospheric Light (background color) for every image. If the water is green, the system automatically adjusts its subtraction values to target green-spectrum haze.")
    add_qa("What are the loss functions used in training?", "We use a composite loss consisting of MSE (Mean Squared Error) for pixel-perfect color matching and SSIM (Structural Similarity) for maintaining the organic textures of fish and corals.")
    add_qa("How do you handle hardware limitations?", "The code features auto-detection. It uses CUDA (NVIDIA GPU) if available for extreme speed, but includes optimized CPU fallbacks for standard laptops.")

    # ---------------------------------------------------------
    # PAGE 10: CONCLUSION & FUTURE SCOPE
    # ---------------------------------------------------------
    pdf.add_page()
    add_section_title("10. CONCLUSION & FUTURE SCOPE")
    add_subheading("10.1 Project Conclusion")
    add_text("This project successfully bridges the gap between traditional marine optics and modern machine learning. By utilizing a hybrid approach, we achieve a system that is not only visually superior but also scientifically explainable. The integration of high-level performance analytics and a premium UI makes this a complete, ready-to-use tool for the marine science community.")
    
    add_subheading("10.2 Future Directions")
    add_text("1. Real-time Video Stream Enhancement: Optimizing the codebase to process live 30FPS feeds from underwater drones.\n2. Marine Species Identification: Integrating a classification head to identify and count fish and coral species automatically after restoration.\n3. Mobile Deployment: Further optimizing the model for mobile devices to be used by divers with underwater camera setups.")

    output_path = "UNDERWATER_AI_FULL_TECHNICAL_REPORT.pdf"
    pdf.output(output_path)
    return output_path

if __name__ == "__main__":
    try:
        path = generate_10_page_report()
        print(f"SUCCESS: Master PDF generated at {os.path.abspath(path)}")
    except Exception as e:
        print(f"ERROR: {str(e)}")
