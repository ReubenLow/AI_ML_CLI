import argparse
from fpdf import FPDF

def generate_pdf(report_type, output_file):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Model {report_type.capitalize()} Report", ln=True, align='C')

    # Assuming metrics are passed via argparse or are available as files, read them
    with open(f"{report_type}_classification_report.txt", "r") as f:
        classification_report = f.read()

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"{report_type.capitalize()} Classification Report", ln=True)
    pdf.multi_cell(0, 10, classification_report)

    # Example metrics for MSE, RMSE, MAE
    train_mse = 0.1234
    train_rmse = 0.2345
    train_mae = 0.1234

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"{report_type.capitalize()} MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}", ln=True)

    pdf.output(output_file)
    print(f"PDF report saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDF report for model performance.")
    parser.add_argument('--report', type=str, help="Path to save the PDF report.")
    parser.add_argument('--type', type=str, help="Type of report (training, validation, random).")

    args = parser.parse_args()
    generate_pdf(args.type, args.report)
