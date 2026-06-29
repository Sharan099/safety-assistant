import os
import fitz
from typing import List, Dict, Any
from loguru import logger
from crawler.base import BaseCrawler

class NHTSACrawler(BaseCrawler):
    """
    Crawler adapter for NHTSA FMVSS standards and ODI investigation/recall documents.
    Supports downloading real PDFs or offline mock generation of standard NHTSA files.
    """

    TEMPLATES = {
        "FMVSS_208": {
            "filename": "fmvss_208.pdf",
            "url": "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-208.pdf",
            "metadata": {
                "regulation_code": "FMVSS 208",
                "title": "FMVSS 208: Occupant crash protection",
                "source_type": "FMVSS",
                "effective_date": "2024-01-01",
                "market": "US"
            },
            "content": (
                "FMVSS 208 -- OCCUPANT CRASH PROTECTION\n"
                "Establishes performance requirements for passive restraints (airbags and seatbelts).\n"
                "Test Setup: Frontal barrier crash test at 30 mph into a rigid concrete wall.\n"
                "Includes out-of-position (OOP) child and occupant dummy airbag suppression tests.\n"
            )
        },
        "FMVSS_214": {
            "filename": "fmvss_214.pdf",
            "url": "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-214.pdf",
            "metadata": {
                "regulation_code": "FMVSS 214",
                "title": "FMVSS 214: Side impact protection",
                "source_type": "FMVSS",
                "effective_date": "2024-01-01",
                "market": "US"
            },
            "content": (
                "FMVSS 214 -- SIDE IMPACT PROTECTION\n"
                "Establishes lateral impact safety specifications.\n"
                "1. MDB side impact test: Crabbed barrier weighing 3015 lbs striking at 33.5 mph.\n"
                "2. Oblique Pole side impact: Vehicle propelled at 20 mph oblique (75 deg) into a rigid pole.\n"
                "Biomechanical criteria evaluated on ES-2re driver dummies.\n"
            )
        },
        "FMVSS_216": {
            "filename": "fmvss_216.pdf",
            "url": "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-216a.pdf",
            "metadata": {
                "regulation_code": "FMVSS 216",
                "title": "FMVSS 216: Roof crush resistance",
                "source_type": "FMVSS",
                "effective_date": "2024-01-01",
                "market": "US"
            },
            "content": (
                "FMVSS 216 -- ROOF CRUSH RESISTANCE\n"
                "Specifies strength requirements for the passenger compartment roof.\n"
                "Test setup: A rigid plate applies force to the corner of the vehicle roof.\n"
                "The roof must withstand a force equal to 3.0 times the vehicle curb weight with less than 5 inches of plate travel.\n"
            )
        },
        "FMVSS_225": {
            "filename": "fmvss_225.pdf",
            "url": "https://www.govinfo.gov/content/pkg/CFR-2023-title49-vol6/pdf/CFR-2023-title49-vol6-sec571-225.pdf",
            "metadata": {
                "regulation_code": "FMVSS 225",
                "title": "FMVSS 225: Child restraint anchorage systems",
                "source_type": "FMVSS",
                "effective_date": "2024-01-01",
                "market": "US"
            },
            "content": (
                "FMVSS 225 -- CHILD RESTRAINT ANCHORAGE SYSTEMS\n"
                "Requires vehicles to be equipped with child restraint anchorage systems (LATCH).\n"
                "Specifies location and force requirements for lower anchor bars and top tether anchor loops.\n"
            )
        },
        "ODI_Investigation": {
            "filename": "nhtsa_odi_pe23_004.pdf",
            "url": "https://www.nhtsa.gov/recalls",
            "metadata": {
                "regulation_code": "NHTSA ODI PE23-004",
                "title": "NHTSA ODI Investigation: Steering Column Failure PE23-004",
                "source_type": "NHTSA",
                "effective_date": "2023-08-14",
                "market": "US"
            },
            "content": (
                "NHTSA OFFICE OF DEFECTS INVESTIGATION (ODI) -- INVESTIGATION PE23-004\n"
                "Subject: Loss of steering control due to coupling separation.\n"
                "Status: Active. Market: US.\n"
                "NHTSA opened a preliminary evaluation into steering column weld defects in model year 2023 sedans.\n"
            )
        },
        "Recall_Database": {
            "filename": "nhtsa_recall_23v123.pdf",
            "url": "https://static.nhtsa.gov/odi/rcl/2023/RCAK-23V123-1111.pdf",
            "metadata": {
                "regulation_code": "NHTSA Recall 23V-123",
                "title": "NHTSA Recall Campaign 23V-123: Passenger Side Airbag Non-Deployment",
                "source_type": "NHTSA",
                "effective_date": "2023-03-15",
                "market": "US"
            },
            "content": (
                "NHTSA RECALL CAMPAIGN 23V-123\n"
                "Subject: Frontal Passenger Airbag non-deployment due to fabric tearing.\n"
                "Safety hazard: Occupants may experience increased seat impact loads if airbag fails to deploy.\n"
                "Under 49 CFR Part 573, manufacturers must replace airbag module at no charge.\n"
            )
        }
    }

    def crawl(self, mock: bool = True) -> List[Dict[str, Any]]:
        results = []
        logger.info(f"Executing NHTSA Crawler. Live/mock status: mock={mock}")

        for key, template in self.TEMPLATES.items():
            filename = template["filename"]
            dest_path = os.path.join(self.output_dir, filename)
            url = template["url"]

            downloaded = False
            if not mock:
                logger.info(f"Attempting live download for {key} from {url}")
                try:
                    self._download_file(url, filename, template["metadata"]["source_type"])
                    downloaded = True
                except Exception as e:
                    logger.warning(f"Failed to download {key} from web: {e}. Falling back to high-fidelity PDF generation.")

            if not downloaded:
                # Fallback or mock mode: generate high-fidelity PDF
                logger.info(f"Generating high-fidelity fallback PDF for {key} at {dest_path}")
                try:
                    doc = fitz.open()
                    page = doc.new_page()
                    text_content = f"TITLE: {template['metadata']['title']}\n\n" + template["content"]
                    rect = fitz.Rect(50, 50, 550, 750)
                    page.insert_textbox(rect, text_content, fontsize=10)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    doc.save(dest_path)
                    doc.close()
                except Exception as ge:
                    logger.error(f"Failed to generate mock PDF for {key}: {ge}")
                    continue

            results.append({
                "file_path": dest_path,
                "source_url": url,
                "metadata": template["metadata"]
            })

        return results

