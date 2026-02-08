from fpdf import FPDF
import os

class MethodologyReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 70, 120)
        self.cell(0, 10, 'Underwater image dehazing using CNN with U-NET: Methodology & Workflow', 0, 1, 'C')
        self.line(20, 18, 190, 18)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Technical Documentation - Page {self.page_no()}", 0, 0, 'C')

def generate_methodology_pdf():
    pdf = MethodologyReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    
    def add_section(title, content):
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 119, 190)
        pdf.multi_cell(170, 10, text=title)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(170, 7, text=content)
        pdf.ln(8)

    pdf.add_page()
    
    # 1. Overview
    add_section("1. METHODOLOGICAL OVERVIEW", 
        "The DeepSea Restoration AI leverages a Hybrid Sequential Architecture that combines deterministic physical modeling with adaptive deep learning. "
        "The process is designed to neutralize the optical degradations of the underwater medium (absorption and scattering) before refining the visual "
        "details using a neural residual network.")

    # 2. Dataset
    add_section("2. DATA ACQUISITION & SMART PRE-PROCESSING", 
        "The system is trained on the UIEB (Underwater Image Enhancement Benchmark) dataset. Our methodology includes:\n"
        "- Recursive Scanning: Automatic discovery of image pairs across deeply nested directory structures.\n"
        "- Zombie Filtering: Detection and exclusion of corrupted or empty (0-byte) image files.\n"
        "- Pre-Calculated Baselines: Every training image undergoes Stage 1 processing to provide the AI with a 'Coarse' baseline for residual learning.")

    # 3. Stage 1 Physics
    add_section("3. STAGE 1: PHYSICS-BASED VARIATIONAL MODELING", 
        "This phase reverses the Light Formation Model: I(x) = J(x)t(x) + B(1-t(x)).\n"
        "- Dark Channel Prior (DCP): Used to estimate depth cues by analyzing the darkest pixels in the scene.\n"
        "- Backlight Estimation (B): Identifies the specific global color of the water (blue, green, or turbid) for targeted subtraction.\n"
        "- Transmission Mapping (t): Calculates a depth heatmap to determine the level of dehazing required per pixel.\n"
        "- Guided Filtering: Refines the transmission map using the raw image as an edge-guide to ensure structural sharpness.")

    # 4. Stage 2 Neural
    add_section("4. STAGE 2: NEURAL RESIDUAL REFINEMENT (U-NET)", 
        "The coarse result is polished by a 23-layer Deep Convolutional Neural Network.\n"
        "- Architecture: A symmetric U-Net featuring 5 levels of encoder-decoder pairing.\n"
        "- Skip Connections: These 'bridges' transfer high-resolution spatial information directly from the encoder to the decoder, "
        "preventing the loss of coral textures or biological features.\n"
        "- Residual Mapping: Instead of generating an image from scratch, the network learns to predict only the corrections needed "
        "for the physics-based Stage 1 output.\n"
        "- Edge-Preserving Smoothing: Implements a high-fidelity Guided Filter post-processing step to achieve pixel clarity without "
        "artificially boosting contrast.")

    # 5. Optimization
    add_section("5. HYBRID FUSION & OPTIMIZATION", 
        "The final image is a fusion: Final = Coarse_Output + AI_Residual.\n"
        "- Loss Functions: We employ a Composite Loss Strategy: MSE (Mean Squared Error) for color precision + SSIM (Structural Similarity) "
        "for textural integrity.\n"
        "- Optimized Aesthetics: The process maintains natural color balances, avoiding aggressive contrast stretching for a premium 'clean' visual look.\n"
        "- Hardware: Automatic detection of CUDA (GPU) or CPU-optimized execution paths.")

    output_path = "UNDERWATER_AI_METHODOLOGY.pdf"
    pdf.output(output_path)
    return output_path

if __name__ == "__main__":
    try:
        path = generate_methodology_pdf()
        print(f"SUCCESS: Methodology PDF generated at {os.path.abspath(path)}")
    except Exception as e:
        print(f"ERROR: {str(e)}")
