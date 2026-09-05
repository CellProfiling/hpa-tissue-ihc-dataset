"""Shared test helpers. Scripts are plain files, not a package: make them importable."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def make_xml(pad_len=0):
    """Minimal HPA XML with one antibody / tissue / patient / image.

    pad_len inserts an XML comment of that many bytes right before <patientId>
    so the tag can be placed across iterparse's 16 KiB read-block boundary.
    """
    pad = f"<!--{'x' * pad_len}-->" if pad_len else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<proteinAtlas>
<entry url="http://www.proteinatlas.org/ENSG00000000001">
<name>GENE1</name>
<identifier db="Ensembl" id="ENSG00000000001"/>
<antibody id="HPA000001">
<tissueExpression assayType="tissue">
<verification type="validation">approved</verification>
<data>
<tissue organ="Soft tissue" ontologyTerms="UBERON_0001013">Adipose tissue</tissue>
<tissueCell><cellType>adipocytes</cellType><level type="staining">medium</level><level type="intensity">moderate</level><quantity>&gt;75%</quantity><location>cytoplasmic/membranous</location></tissueCell>
<patient><sex>Male</sex><age>44</age>{pad}<patientId>4016</patientId>
<sample><snomedParameters><snomed tissueDescription="Normal tissue, NOS" snomedCode="M-00100"/></snomedParameters>
<assayImage><image imageType="selected"><imageUrl>http://images.proteinatlas.org/1/100007_A_1_8.jpg</imageUrl><imageUrlTif>http://images.proteinatlas.org/1/100007_A_1_8.tif</imageUrlTif></image></assayImage></sample>
</patient>
</data>
</tissueExpression>
</antibody>
</entry>
</proteinAtlas>
"""
