"""Script to generate a realistic SEC 10-K PDF sample with multi-column financial tables."""

import hashlib
import json
from pathlib import Path
import fitz  # PyMuPDF


def create_nvidia_10k_pdf(output_path: Path) -> str:
    """Create a realistic 3-page SEC 10-K document and return its SHA-256 hash."""
    doc = fitz.open()

    # --- Page 1: Cover & Business ---
    page1 = doc.new_page(width=612, height=792)  # Standard Letter size
    p1_text = """UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-K
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
For the fiscal year ended January 28, 2024

NVIDIA CORPORATION

PART I

Item 1. Business
NVIDIA pioneered accelerated computing to help solve the most challenging computational problems. Since our founding in 1993, we have specialized in graphics processing units, or GPUs. Our inventions have helped drive modern artificial intelligence, high performance computing, and computer graphics.

Our compute architecture integrates hardware and software, combining our GPUs, central processing units, or CPUs, data processing units, or DPUs, interconnects, and our CUDA programming model. Our computing platforms power modern generative AI workloads across hyperscale cloud providers and enterprise data centers worldwide. Key products include the NVIDIA Hopper architecture H100 Tensor Core GPU, Grace CPU, and Quantum-2 InfiniBand networking platforms.
"""
    page1.insert_text((54, 54), p1_text, fontsize=10, fontname="helv")
    page1.insert_text((54, 750), "Page 1", fontsize=9, fontname="helv")

    # --- Page 2: MD&A and Segment Revenue Table ---
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((54, 36), "NVIDIA CORPORATION | 2024 ANNUAL REPORT", fontsize=8, fontname="helv")

    p2_heading = """PART II

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations

Revenue by Reportable Segment
Our business is organized into two primary reportable segments: Compute & Networking and Graphics. We also monitor revenue across specialized market platforms: Data Center, Gaming, Professional Visualization, Automotive, and OEM and Other.

The following table summarizes our revenue by market platform for fiscal years 2024 and 2023:
"""
    page2.insert_text((54, 60), p2_heading, fontsize=10, fontname="helv")

    # Draw table border and text for Segment Revenue
    table_text = """+---------------------------+-------------------+-------------------+----------+
| Market Platform           | Fiscal 2024 ($M)  | Fiscal 2023 ($M)  | % Change |
+---------------------------+-------------------+-------------------+----------+
| Data Center               | 47,525            | 15,013            | +217%    |
| Gaming                    | 10,447            | 9,067             | +15%     |
| Professional Visualization| 1,553             | 1,544             | +1%      |
| Automotive                | 1,091             | 903               | +21%     |
| OEM and Other             | 306               | 447               | -32%     |
| Total Revenue             | 60,922            | 26,974            | +126%    |
+---------------------------+-------------------+-------------------+----------+"""

    page2.insert_text((54, 230), table_text, fontsize=9, fontname="courier")

    p2_notes = """* Data Center revenue rose 217% year-over-year to $47.5 billion, primarily driven by strong enterprise demand for NVIDIA HGX platforms built on Hopper architecture GPUs for large language models, recommendation engines, and generative AI applications.
* Gaming revenue rose 15% to $10.4 billion, reflecting strong customer adoption of our GeForce RTX 40 Series desktop and notebook GPUs featuring Ada Lovelace architecture."""

    page2.insert_text((54, 380), p2_notes, fontsize=10, fontname="helv")
    page2.insert_text((54, 750), "Page 45", fontsize=9, fontname="helv")

    # --- Page 3: Consolidated Statements of Income ---
    page3 = doc.new_page(width=612, height=792)
    page3.insert_text((54, 36), "NVIDIA CORPORATION | 2024 ANNUAL REPORT", fontsize=8, fontname="helv")

    p3_heading = """Consolidated Statements of Income
(In millions, except per share data)

The following financial statement reflects our consolidated operations for the fiscal years ended January 28, 2024, January 29, 2023, and January 30, 2022:
"""
    page3.insert_text((54, 60), p3_heading, fontsize=10, fontname="helv")

    income_table = """+--------------------------------+------------+------------+------------+
| Operating Statement Line Item  | 2024 ($M)  | 2023 ($M)  | 2022 ($M)  |
+--------------------------------+------------+------------+------------+
| Revenue                        | 60,922     | 26,974     | 26,914     |
| Cost of revenue                | 16,621     | 11,618     | 10,689     |
| Gross profit                   | 44,301     | 15,356     | 16,225     |
| Research and development       | 8,675      | 7,339      | 5,268      |
| Sales, general and admin       | 2,654      | 2,440      | 2,166      |
| Total operating expenses       | 11,329     | 9,779      | 7,434      |
| Operating income               | 32,972     | 5,577      | 8,791      |
| Net income                     | 29,760     | 4,368      | 9,752      |
| Diluted net income per share   | $11.93     | $1.74      | $3.85      |
+--------------------------------+------------+------------+------------+"""

    page3.insert_text((54, 150), income_table, fontsize=9, fontname="courier")

    p3_notes = """Operating income increased 491% in fiscal year 2024 compared to fiscal year 2023. Gross margin increased to 72.7% from 56.9%, primarily reflecting favorable component costs and higher Data Center compute revenue contribution. Diluted earnings per share reached $11.93 compared to $1.74 in fiscal 2023."""

    page3.insert_text((54, 340), p3_notes, fontsize=10, fontname="helv")
    page3.insert_text((54, 750), "Page 46", fontsize=9, fontname="helv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()

    # Calculate SHA-256
    with open(output_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    return file_hash


if __name__ == "__main__":
    out_pdf = Path("data/raw_corpus/nvidia_10k_sample.pdf")
    sha256 = create_nvidia_10k_pdf(out_pdf)
    print(f"Generated {out_pdf} with SHA-256: {sha256}")

    manifest = {
        "domain": "SEC Financial Filings (10-K / 10-Q)",
        "description": "Curated enterprise financial documents with multi-column balance sheets and segment disclosures.",
        "documents": [
            {
                "filename": "nvidia_10k_sample.pdf",
                "sha256": sha256,
                "title": "NVIDIA Corporation 2024 Form 10-K Annual Report",
                "total_pages": 3,
                "category": "financial_report",
                "challenging_elements": [
                    "Multi-column ASCII/Unicode tabular layout on Page 2 and Page 3",
                    "Dense numerical financial metrics across fiscal years 2022-2024",
                    "Repeating page headers and footers with page numbering",
                    "Footnote disclosures directly tied to segment revenue rows"
                ]
            }
        ]
    }

    manifest_path = Path("data/raw_corpus/manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest to {manifest_path}")
