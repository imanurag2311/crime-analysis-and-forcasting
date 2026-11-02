"""
IPC 1860 → BNS 2023 Crosswalk
================================
Centralized statutory crosswalk mapping Indian Penal Code (IPC) 1860 sections
to Bharatiya Nyaya Sanhita (BNS) 2023 sections.

Also classifies crime domains and standardizes crime category names.
"""

import logging
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# IPC → BNS STATUTORY MAPPING
# ============================================================

IPC_TO_BNS_MAP: Dict[str, Dict[str, str]] = {
    # --- Violent Crimes ---
    "IPC 302": {
        "bns": "BNS 103",
        "crime": "Murder",
        "domain": "Violent Crime"
    },
    "IPC 304": {
        "bns": "BNS 105",
        "crime": "Culpable Homicide not amounting to Murder",
        "domain": "Violent Crime"
    },
    "IPC 304A": {
        "bns": "BNS 106",
        "crime": "Death by Negligence",
        "domain": "Violent Crime"
    },
    "IPC 304B": {
        "bns": "BNS 80",
        "crime": "Dowry Death",
        "domain": "Crimes Against Women (CAW)"
    },
    "IPC 306": {
        "bns": "BNS 108",
        "crime": "Abetment to Suicide",
        "domain": "Violent Crime"
    },
    "IPC 307": {
        "bns": "BNS 109",
        "crime": "Attempt to Murder",
        "domain": "Violent Crime"
    },
    "IPC 308": {
        "bns": "BNS 110",
        "crime": "Attempt to Culpable Homicide",
        "domain": "Violent Crime"
    },
    "IPC 323": {
        "bns": "BNS 115",
        "crime": "Voluntarily Causing Hurt",
        "domain": "General IPC"
    },
    "IPC 324": {
        "bns": "BNS 118",
        "crime": "Voluntarily Causing Hurt by Dangerous Weapons",
        "domain": "General IPC"
    },
    "IPC 325": {
        "bns": "BNS 117",
        "crime": "Voluntarily Causing Grievous Hurt",
        "domain": "Violent Crime"
    },
    "IPC 326": {
        "bns": "BNS 118",
        "crime": "Voluntarily Causing Grievous Hurt by Dangerous Weapons",
        "domain": "Violent Crime"
    },

    # --- Crimes Against Women ---
    "IPC 354": {
        "bns": "BNS 74",
        "crime": "Molestation / Assault on Woman",
        "domain": "Crimes Against Women (CAW)"
    },
    "IPC 354A": {
        "bns": "BNS 75",
        "crime": "Sexual Harassment",
        "domain": "Crimes Against Women (CAW)"
    },
    "IPC 354B": {
        "bns": "BNS 76",
        "crime": "Assault with Intent to Disrobe",
        "domain": "Crimes Against Women (CAW)"
    },
    "IPC 354C": {
        "bns": "BNS 77",
        "crime": "Voyeurism",
        "domain": "Crimes Against Women (CAW)"
    },
    "IPC 354D": {
        "bns": "BNS 78",
        "crime": "Stalking",
        "domain": "Crimes Against Women (CAW)"
    },
    "IPC 363": {
        "bns": "BNS 137",
        "crime": "Kidnapping",
        "domain": "General IPC"
    },
    "IPC 366": {
        "bns": "BNS 139",
        "crime": "Kidnapping / Abducting Woman",
        "domain": "Crimes Against Women (CAW)"
    },
    "IPC 376": {
        "bns": "BNS 64",
        "crime": "Rape",
        "domain": "Crimes Against Women (CAW)"
    },
    "IPC 376A": {
        "bns": "BNS 66",
        "crime": "Rape causing Death or Vegetative State",
        "domain": "Crimes Against Women (CAW)"
    },
    "IPC 376D": {
        "bns": "BNS 70",
        "crime": "Gang Rape",
        "domain": "Crimes Against Women (CAW)"
    },
    "IPC 498A": {
        "bns": "BNS 85",
        "crime": "Cruelty by Husband / Relatives (Domestic Violence)",
        "domain": "Crimes Against Women (CAW)"
    },
    "IPC 509": {
        "bns": "BNS 79",
        "crime": "Word / Gesture / Act to Insult Modesty of Woman",
        "domain": "Crimes Against Women (CAW)"
    },

    # --- Property Crimes ---
    "IPC 379": {
        "bns": "BNS 303",
        "crime": "Theft",
        "domain": "Property Crime"
    },
    "IPC 380": {
        "bns": "BNS 305",
        "crime": "Theft in Dwelling House",
        "domain": "Property Crime"
    },
    "IPC 381": {
        "bns": "BNS 305",
        "crime": "Theft by Clerk or Servant",
        "domain": "Property Crime"
    },
    "IPC 382": {
        "bns": "BNS 304",
        "crime": "Theft with Preparation for Death / Hurt / Restraint",
        "domain": "Property Crime"
    },
    "IPC 392": {
        "bns": "BNS 309",
        "crime": "Robbery",
        "domain": "Property Crime"
    },
    "IPC 393": {
        "bns": "BNS 309",
        "crime": "Attempt to Commit Robbery",
        "domain": "Property Crime"
    },
    "IPC 394": {
        "bns": "BNS 310",
        "crime": "Voluntarily Causing Hurt in Committing Robbery",
        "domain": "Property Crime"
    },
    "IPC 395": {
        "bns": "BNS 310",
        "crime": "Dacoity",
        "domain": "Property Crime"
    },
    "IPC 397": {
        "bns": "BNS 312",
        "crime": "Robbery / Dacoity with Attempt to Cause Death / Grievous Hurt",
        "domain": "Property Crime"
    },
    "IPC 399": {
        "bns": "BNS 313",
        "crime": "Making Preparation for Dacoity",
        "domain": "Property Crime"
    },
    "IPC 411": {
        "bns": "BNS 317",
        "crime": "Dishonestly Receiving Stolen Property",
        "domain": "Property Crime"
    },
    "IPC 454": {
        "bns": "BNS 331",
        "crime": "Lurking House-Trespass / House-breaking (Concealment)",
        "domain": "Property Crime"
    },
    "IPC 457": {
        "bns": "BNS 333",
        "crime": "Lurking House-Trespass / House-breaking at Night",
        "domain": "Property Crime"
    },

    # --- Economic Offences ---
    "IPC 406": {
        "bns": "BNS 316",
        "crime": "Criminal Breach of Trust",
        "domain": "Economic Offences (EOW)"
    },
    "IPC 420": {
        "bns": "BNS 318",
        "crime": "Cheating / Fraud",
        "domain": "Economic Offences (EOW)"
    },
    "IPC 467": {
        "bns": "BNS 338",
        "crime": "Forgery of Valuable Security",
        "domain": "Economic Offences (EOW)"
    },
    "IPC 468": {
        "bns": "BNS 339",
        "crime": "Forgery for Purpose of Cheating",
        "domain": "Economic Offences (EOW)"
    },
    "IPC 471": {
        "bns": "BNS 340",
        "crime": "Using Forged Document as Genuine",
        "domain": "Economic Offences (EOW)"
    },

    # --- Other Crimes ---
    "IPC 143": {
        "bns": "BNS 189",
        "crime": "Unlawful Assembly",
        "domain": "General IPC"
    },
    "IPC 147": {
        "bns": "BNS 191",
        "crime": "Rioting",
        "domain": "General IPC"
    },
    "IPC 148": {
        "bns": "BNS 192",
        "crime": "Rioting Armed with Deadly Weapon",
        "domain": "General IPC"
    },
    "IPC 279": {
        "bns": "BNS 281",
        "crime": "Rash Driving on Public Way",
        "domain": "General IPC"
    },
    "IPC 336": {
        "bns": "BNS 125",
        "crime": "Act Endangering Life / Personal Safety",
        "domain": "General IPC"
    },
    "IPC 353": {
        "bns": "BNS 132",
        "crime": "Assault / Criminal Force to Deter Public Servant",
        "domain": "General IPC"
    },
    "IPC 365": {
        "bns": "BNS 140",
        "crime": "Kidnapping with Intent to Secretly / Wrongfully Confine",
        "domain": "General IPC"
    },
    "IPC 427": {
        "bns": "BNS 324",
        "crime": "Mischief Causing Damage",
        "domain": "General IPC"
    },
    "IPC 452": {
        "bns": "BNS 329",
        "crime": "House-Trespass with Preparation for Hurt / Assault",
        "domain": "General IPC"
    },
    "IPC 504": {
        "bns": "BNS 351",
        "crime": "Intentional Insult with Intent to Provoke Breach of Peace",
        "domain": "General IPC"
    },
    "IPC 506": {
        "bns": "BNS 351",
        "crime": "Criminal Intimidation",
        "domain": "General IPC"
    },

    # --- Special / Local Laws (not strictly IPC but commonly encountered) ---
    "NDPS Act": {
        "bns": "NDPS Act",
        "crime": "Narcotic Drugs and Psychotropic Substances",
        "domain": "NDPS Act"
    },
    "Arms Act": {
        "bns": "Arms Act",
        "crime": "Arms Act Violations",
        "domain": "General IPC"
    },
    "IT Act 66": {
        "bns": "IT Act 66",
        "crime": "Computer Related Offences / Cyber Crime",
        "domain": "Cyber Crime"
    },
}

# Build reverse map (BNS → IPC)
BNS_TO_IPC_MAP: Dict[str, Dict[str, str]] = {}
for ipc_key, mapping in IPC_TO_BNS_MAP.items():
    bns_key = mapping["bns"]
    if bns_key not in BNS_TO_IPC_MAP:
        BNS_TO_IPC_MAP[bns_key] = {
            "ipc": ipc_key,
            "crime": mapping["crime"],
            "domain": mapping["domain"]
        }


# ============================================================
# CRIME CATEGORY STANDARDIZATION MAP
# ============================================================

CATEGORY_ALIASES: Dict[str, str] = {
    # Crimes Against Women variations
    "caw rape": "CAW_Rape",
    "c.a.w. rape": "CAW_Rape",
    "c a w rape": "CAW_Rape",
    "crime against women - rape": "CAW_Rape",
    "crime against women rape": "CAW_Rape",
    "crimes against women - rape": "CAW_Rape",
    "rape": "CAW_Rape",

    "caw molestation": "CAW_Molestation",
    "c.a.w. molestation": "CAW_Molestation",
    "molestation": "CAW_Molestation",
    "assault on women": "CAW_Molestation",
    "crime against women - molestation": "CAW_Molestation",

    "caw dowry death": "CAW_Dowry_Death",
    "dowry death": "CAW_Dowry_Death",
    "dowry deaths": "CAW_Dowry_Death",

    "caw domestic violence": "CAW_Domestic_Violence",
    "domestic violence": "CAW_Domestic_Violence",
    "cruelty by husband": "CAW_Domestic_Violence",
    "498a": "CAW_Domestic_Violence",

    "caw sexual harassment": "CAW_Sexual_Harassment",
    "sexual harassment": "CAW_Sexual_Harassment",

    "caw kidnapping": "CAW_Kidnapping",
    "kidnapping of women": "CAW_Kidnapping",
    "abduction of women": "CAW_Kidnapping",

    "caw stalking": "CAW_Stalking",
    "stalking": "CAW_Stalking",

    # Property crimes
    "theft": "Theft",
    "thefts": "Theft",
    "thefts.": "Theft",
    "theft.": "Theft",
    "house theft": "House_Theft",
    "house breaking": "House_Breaking",
    "house-breaking": "House_Breaking",
    "housebreaking": "House_Breaking",

    "robbery": "Robbery",
    "robberies": "Robbery",
    "dacoity": "Dacoity",
    "dacoities": "Dacoity",

    "vehicle theft": "Vehicle_Theft",
    "motor vehicle theft": "Vehicle_Theft",
    "auto theft": "Vehicle_Theft",

    "chain snatching": "Chain_Snatching",
    "chain-snatching": "Chain_Snatching",

    # Violent crimes
    "murder": "Murder",
    "murders": "Murder",
    "attempt to murder": "Attempt_to_Murder",
    "attempt murder": "Attempt_to_Murder",
    "att. to murder": "Attempt_to_Murder",

    "culpable homicide": "Culpable_Homicide",
    "death by negligence": "Death_by_Negligence",
    "accidental death": "Accidental_Death",
    "abetment to suicide": "Abetment_to_Suicide",

    "rioting": "Rioting",
    "riots": "Rioting",

    "kidnapping": "Kidnapping",
    "kidnapping & abduction": "Kidnapping",
    "kidnapping and abduction": "Kidnapping",

    "assault": "Assault",
    "causing hurt": "Causing_Hurt",
    "grievous hurt": "Grievous_Hurt",

    # Economic offences
    "cheating": "Cheating_Fraud",
    "fraud": "Cheating_Fraud",
    "cheating & fraud": "Cheating_Fraud",
    "cheating and fraud": "Cheating_Fraud",
    "criminal breach of trust": "Criminal_Breach_of_Trust",
    "forgery": "Forgery",
    "counterfeiting": "Counterfeiting",

    # NDPS
    "ndps": "NDPS",
    "ndps act": "NDPS",
    "narcotic drugs": "NDPS",
    "drug offence": "NDPS",
    "drug offences": "NDPS",

    # Cyber crime
    "cyber crime": "Cyber_Crime",
    "cybercrime": "Cyber_Crime",
    "cyber crimes": "Cyber_Crime",
    "online fraud": "Cyber_Crime",
    "it act": "Cyber_Crime",

    # Others
    "rash driving": "Rash_Driving",
    "arms act": "Arms_Act",
    "prohibition act": "Prohibition_Act",
    "gambling act": "Gambling_Act",
    "missing person": "Missing_Person",
    "missing persons": "Missing_Person",
    "unnatural death": "Unnatural_Death",
}


CATEGORY_DISPLAY_NAMES: Dict[str, str] = {
    "Att.To.C.Murder": "Attempt to Commit Murder",
    "Att.To.C.Robbery": "Attempt to Commit Robbery",
    "CAW_Molestation": "Crimes Against Women - Molestation",
    "CAW_Rape": "Crimes Against Women - Rape",
    "CAW_Dowry_Death": "Crimes Against Women - Dowry Death",
    "CAW_Domestic_Violence": "Crimes Against Women - Domestic Violence",
    "CAW_Sexual_Harassment": "Crimes Against Women - Sexual Harassment",
    "CAW_Kidnapping": "Crimes Against Women - Kidnapping & Abduction",
    "CAW_Stalking": "Crimes Against Women - Stalking",
    "Dacoity": "Dacoity",
    "Extortion": "Extortion",
    "H.B.T.Day": "House Breaking (Day)",
    "H.B.T.Day_Night": "House Breaking (Day & Night)",
    "H.B.T.Night.": "House Breaking (Night)",
    "Hurt": "Voluntarily Causing Hurt",
    "M.V.Thefts.": "Motor Vehicle Theft",
    "Molestaion": "Molestation",
    "Murder": "Murder",
    "Other_I.P.C.": "Other IPC Crimes",
    "Pocket_Picking": "Pickpocketing",
    "Prep.For_Dacoity": "Preparation for Dacoity",
    "Riots.": "Rioting & Unlawful Assembly",
    "Robbery": "Robbery",
    "Robbery_(Excluding_Chain_Snatching)": "Robbery (Excl. Chain Snatching)",
    "Robbery_(Excluding_Chain_Snatching)95": "Robbery (Sec 395/397 Variant)",
    "Robbery_Chain_Snatching": "Robbery - Chain Snatching",
    "Sexual_Offences_(Sec_69_Bns_Only_Mejor)": "Sexual Offences (Sec 69 BNS)",
    "Snatching": "Snatching",
    "Theft": "Theft",
    "Thefts.": "Theft",
}

CATEGORY_LEGAL_MAP: Dict[str, Dict[str, str]] = {
    "Att.To.C.Murder": {"ipc": "IPC 307", "bns": "BNS 109", "domain": "Violent Crime"},
    "Att.To.C.Robbery": {"ipc": "IPC 393", "bns": "BNS 309", "domain": "Property Crime"},
    "CAW_Molestation": {"ipc": "IPC 354", "bns": "BNS 74", "domain": "Crimes Against Women (CAW)"},
    "CAW_Rape": {"ipc": "IPC 376", "bns": "BNS 64", "domain": "Crimes Against Women (CAW)"},
    "CAW_Dowry_Death": {"ipc": "IPC 304B", "bns": "BNS 80", "domain": "Crimes Against Women (CAW)"},
    "CAW_Domestic_Violence": {"ipc": "IPC 498A", "bns": "BNS 85", "domain": "Crimes Against Women (CAW)"},
    "CAW_Sexual_Harassment": {"ipc": "IPC 354A", "bns": "BNS 75", "domain": "Crimes Against Women (CAW)"},
    "CAW_Kidnapping": {"ipc": "IPC 366", "bns": "BNS 139", "domain": "Crimes Against Women (CAW)"},
    "CAW_Stalking": {"ipc": "IPC 354D", "bns": "BNS 78", "domain": "Crimes Against Women (CAW)"},
    "Dacoity": {"ipc": "IPC 395", "bns": "BNS 310", "domain": "Property Crime"},
    "Extortion": {"ipc": "IPC 384", "bns": "BNS 308", "domain": "Property Crime"},
    "H.B.T.Day": {"ipc": "IPC 454", "bns": "BNS 331", "domain": "Property Crime"},
    "H.B.T.Day_Night": {"ipc": "IPC 457", "bns": "BNS 333", "domain": "Property Crime"},
    "H.B.T.Night.": {"ipc": "IPC 457", "bns": "BNS 333", "domain": "Property Crime"},
    "Hurt": {"ipc": "IPC 323", "bns": "BNS 115", "domain": "General IPC"},
    "M.V.Thefts.": {"ipc": "IPC 379", "bns": "BNS 303", "domain": "Property Crime"},
    "Molestaion": {"ipc": "IPC 354", "bns": "BNS 74", "domain": "Crimes Against Women (CAW)"},
    "Murder": {"ipc": "IPC 302", "bns": "BNS 103", "domain": "Violent Crime"},
    "Other_I.P.C.": {"ipc": "IPC 504", "bns": "BNS 351", "domain": "General IPC"},
    "Pocket_Picking": {"ipc": "IPC 379", "bns": "BNS 303", "domain": "Property Crime"},
    "Prep.For_Dacoity": {"ipc": "IPC 399", "bns": "BNS 313", "domain": "Property Crime"},
    "Riots.": {"ipc": "IPC 147", "bns": "BNS 191", "domain": "General IPC"},
    "Robbery": {"ipc": "IPC 392", "bns": "BNS 309", "domain": "Property Crime"},
    "Robbery_(Excluding_Chain_Snatching)": {"ipc": "IPC 392", "bns": "BNS 309", "domain": "Property Crime"},
    "Robbery_(Excluding_Chain_Snatching)95": {"ipc": "IPC 395", "bns": "BNS 310", "domain": "Property Crime"},
    "Robbery_Chain_Snatching": {"ipc": "IPC 392", "bns": "BNS 304", "domain": "Property Crime"},
    "Sexual_Offences_(Sec_69_Bns_Only_Mejor)": {"ipc": "IPC 376", "bns": "BNS 69", "domain": "Crimes Against Women (CAW)"},
    "Snatching": {"ipc": "IPC 379", "bns": "BNS 304", "domain": "Property Crime"},
    "Theft": {"ipc": "IPC 379", "bns": "BNS 303", "domain": "Property Crime"},
    "Total_Ipc": {"ipc": "IPC Summary", "bns": "BNS Summary", "domain": "General IPC"},
}


class IPCBNSCrosswalk:
    """
    IPC ↔ BNS crosswalk handler.
    
    Provides:
        - IPC → BNS lookup
        - BNS → IPC reverse lookup
        - Crime category standardization
        - Domain classification
        - Category display name formatting
        - Category legal sections fallback lookup
    """

    def __init__(self):
        """Initialize with the statutory mapping."""
        self.ipc_to_bns = IPC_TO_BNS_MAP
        self.bns_to_ipc = BNS_TO_IPC_MAP
        self.category_aliases = CATEGORY_ALIASES
        self.display_names = CATEGORY_DISPLAY_NAMES
        self.category_legal_map = CATEGORY_LEGAL_MAP
        logger.info(
            f"IPC/BNS crosswalk initialized: {len(self.ipc_to_bns)} IPC mappings"
        )

    def get_category_legal_info(self, category: str) -> Dict[str, str]:
        """Get default IPC section, BNS section, and domain for a category."""
        if not category:
            return {"ipc": "", "bns": "", "domain": "Other"}
        return self.category_legal_map.get(category, {"ipc": "", "bns": "", "domain": "Other"})

    def get_display_name(self, category: str) -> str:
        """
        Get full human-readable display name for a category.
        e.g., 'Att.To.C.Murder' → 'Attempt to Commit Murder'
        """
        if not category:
            return ""
        if category in self.display_names:
            return self.display_names[category]
        # Fallback: clean up underscores and dots
        clean = category.replace("_", " ").replace(".", " ").strip()
        while "  " in clean:
            clean = clean.replace("  ", " ")
        return clean

    def lookup_ipc(self, ipc_section: str) -> Optional[Dict[str, str]]:
        """
        Look up BNS equivalent for an IPC section.
        
        Args:
            ipc_section: IPC section string (e.g., "IPC 302", "302")
            
        Returns:
            Dict with 'bns', 'crime', 'domain' or None
        """
        # Normalize input
        normalized = self._normalize_section(ipc_section, prefix="IPC")

        if normalized in self.ipc_to_bns:
            return self.ipc_to_bns[normalized]

        # Try without prefix
        for key, value in self.ipc_to_bns.items():
            if key.replace("IPC ", "") == ipc_section.strip():
                return value

        logger.debug(f"No BNS mapping found for: {ipc_section}")
        return None

    def lookup_bns(self, bns_section: str) -> Optional[Dict[str, str]]:
        """
        Reverse lookup: BNS → IPC.
        
        Args:
            bns_section: BNS section string (e.g., "BNS 103", "103")
            
        Returns:
            Dict with 'ipc', 'crime', 'domain' or None
        """
        normalized = self._normalize_section(bns_section, prefix="BNS")

        if normalized in self.bns_to_ipc:
            return self.bns_to_ipc[normalized]

        for key, value in self.bns_to_ipc.items():
            if key.replace("BNS ", "") == bns_section.strip():
                return value

        logger.debug(f"No IPC mapping found for: {bns_section}")
        return None

    def _normalize_section(self, section: str, prefix: str = "IPC") -> str:
        """Normalize section string (e.g., '302' → 'IPC 302')."""
        section = section.strip().upper()
        if not section.startswith(prefix):
            # Remove any existing prefix
            section = section.replace("IPC ", "").replace("BNS ", "")
            section = f"{prefix} {section}"
        return section

    def get_bns_section(self, ipc_section: str) -> str:
        """Get BNS section string for an IPC section. Returns empty if no mapping."""
        result = self.lookup_ipc(ipc_section)
        return result["bns"] if result else ""

    def get_domain(self, ipc_section: str) -> str:
        """Get domain classification for an IPC section."""
        result = self.lookup_ipc(ipc_section)
        return result["domain"] if result else "Other"

    def get_crime_name(self, ipc_section: str) -> str:
        """Get standardized crime name for an IPC section."""
        result = self.lookup_ipc(ipc_section)
        return result["crime"] if result else ""

    def standardize_category(self, raw_category: str) -> str:
        """
        Standardize a raw crime category name into the canonical form.
        
        Uses the CATEGORY_ALIASES mapping for normalization.
        If no alias found, applies basic normalization.
        """
        if not raw_category:
            return "Unknown"

        lower = raw_category.strip().lower()

        # Check aliases
        if lower in self.category_aliases:
            return self.category_aliases[lower]

        # Basic normalization: Title case, replace spaces with underscores
        normalized = raw_category.strip().title().replace(" ", "_")
        normalized = normalized.replace("-", "_").replace("/", "_")
        # Remove consecutive underscores
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized

    def apply_crosswalk(self, legal_section: str) -> Dict[str, str]:
        """
        Apply the full crosswalk for a given legal section.
        
        Returns dict with:
            - bns_section
            - domain
            - crime_name
            - standard_group
        """
        bns = self.get_bns_section(legal_section)
        domain = self.get_domain(legal_section)
        crime_name = self.get_crime_name(legal_section)

        return {
            "legal_section_bns": bns,
            "domain": domain,
            "crime_name": crime_name,
            "standard_group": domain
        }

    def get_full_mapping_table(self) -> List[Dict[str, str]]:
        """
        Get the complete IPC ↔ BNS mapping as a list of dicts.
        Useful for display in the dashboard.
        """
        table = []
        for ipc_key, mapping in self.ipc_to_bns.items():
            table.append({
                "IPC Section": ipc_key,
                "Crime": mapping["crime"],
                "BNS Section": mapping["bns"],
                "Domain": mapping["domain"]
            })
        return table

    def search_mapping(self, query: str) -> List[Dict[str, str]]:
        """
        Search the crosswalk mapping by query string.
        Searches across IPC sections, BNS sections, crime names, and domains.
        """
        query_lower = query.lower().strip()
        results = []

        for ipc_key, mapping in self.ipc_to_bns.items():
            if (query_lower in ipc_key.lower() or
                    query_lower in mapping["bns"].lower() or
                    query_lower in mapping["crime"].lower() or
                    query_lower in mapping["domain"].lower()):
                results.append({
                    "IPC Section": ipc_key,
                    "Crime": mapping["crime"],
                    "BNS Section": mapping["bns"],
                    "Domain": mapping["domain"]
                })

        return results
