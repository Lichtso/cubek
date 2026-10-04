use crate::{ScanKind, instructions::ScanInstructionKind};
use cubek_test_utils::CatalogEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanProblem {
    pub scan_kind: ScanKind,
    pub scan_instruction_kind: ScanInstructionKind,
    pub size: usize,
}

const SIZES: &[usize] = &[10_000_000];

pub fn problems() -> Vec<CatalogEntry<ScanProblem>> {
    let mut entries = Vec::new();

    for &size in SIZES {
        for &scan_instruction_kind in ScanInstructionKind::ALL {
            for &scan_kind in ScanKind::ALL {
                entries.push(CatalogEntry::new(
                    format!(
                        "{}_{}_{}",
                        scan_kind.label().to_lowercase(),
                        scan_instruction_kind.label().to_lowercase(),
                        size,
                    ),
                    format!(
                        "{} {} over {} elements",
                        scan_kind.label(),
                        scan_instruction_kind.label(),
                        size,
                    ),
                    ScanProblem {
                        scan_kind,
                        scan_instruction_kind,
                        size,
                    },
                ));
            }
        }
    }

    entries
}
