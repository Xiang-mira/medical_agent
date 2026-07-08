import os
import math
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import (binary_dilation as scipy_dilation,
                           label as cc_label,
                           generate_binary_structure,
                           distance_transform_edt,
                           zoom)
import warnings
warnings.filterwarnings('ignore')

# ── HU windows (empirically derived from PanTS training set) ─────────────────
# Only defined for organs that receive HU-gated dilation (hollow + colon).
# Solid organs and tubular organs do not use HU windows.
#
# Gallbladder: bile 0-40 HU (near-water density)
# Duodenum:    duodenal wall soft tissue 20-60 HU
# Stomach:     wall and fluid contents -50 to 100 HU
# Intestine:   wall and contents -50 to 100 HU
# Adrenal:     soft tissue 20-50 HU

HU_WINDOWS = {
    'gallbladder': ( 0,  40),
    'duodenum':    (20,  60),
    'stomach':     (-50, 100),
    'intestine':   (-50, 100),
    'adrenal_l':   (20,  50),
    'adrenal_r':   (20,  50),
}

HU_AIR_RANGE      = (-1024, -350)
COLON_GAS_DIST_MM = 5.0

# FragRatio thresholds — only correct solid/tubular organs below this value
FRAGRATIO_THRESHOLDS = {
    'liver':               0.95,
    'spleen':              0.93,
    'kidney_l':            0.92,
    'kidney_r':            0.92,
    'pancreas':            0.88,
    'aorta':               0.85,
    'ivc':                 0.85,
    'portal_splenic_vein': 0.80,
    'renal_vein_l':        0.80,
    'renal_vein_r':        0.80,
    'celiac_artery':       0.80,
    'sma':                 0.80,
}

# Dijkstra gap thresholds in mm — fragments further apart are left separate
DIJKSTRA_GAP_MM = {
    'aorta':               30,
    'ivc':                 30,
    'portal_splenic_vein': 25,
    'renal_vein_l':        20,
    'renal_vein_r':        20,
    'celiac_artery':       20,
    'sma':                 20,
}

# organ categories
SOLID_ORGANS   = ['liver', 'spleen', 'kidney_l', 'kidney_r', 'pancreas']
HOLLOW_ORGANS  = ['gallbladder', 'duodenum', 'stomach', 'adrenal_l', 'adrenal_r', 'bladder']
COLON_ORGANS   = ['colon', 'intestine']
TUBULAR_ORGANS = ['aorta', 'ivc', 'portal_splenic_vein', 'renal_vein_l', 'renal_vein_r',
                  'celiac_artery', 'sma']


# ── helpers ───────────────────────────────────────────────────────────────────

def frag_ratio(mask):
    if mask.sum() == 0:
        return 1.0
    labeled, n = cc_label(mask)
    if n <= 1:
        return 1.0
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    return float(max(sizes)) / float(mask.sum())


def largest_component(mask):
    if mask.sum() == 0:
        return mask
    labeled, n = cc_label(mask)
    if n <= 1:
        return mask.astype(np.uint8)
    sizes  = sorted(range(1, n + 1), key=lambda i: (labeled == i).sum(), reverse=True)
    result = np.zeros_like(mask)
    result[labeled == sizes[0]] = 1
    return result.astype(np.uint8)


def remove_noise(mask, min_voxels=50):
    """Remove spurious small disconnected components below min_voxels threshold."""
    if mask.sum() == 0:
        return mask
    labeled, num = cc_label(mask)
    result = np.zeros_like(mask)
    for i in range(1, num + 1):
        if (labeled == i).sum() >= min_voxels:
            result[labeled == i] = 1
    return result.astype(np.uint8)


# ── correction 1: largest component for solid organs ─────────────────────────
# SSM MAP refinement (paper Eq. 1-2) requires PanTS training data.
# Largest component is used here as the fallback.

def correct_solid(mask, organ, voxel_spacing=(1.0, 1.0, 1.0)):
    if mask.sum() == 0:
        return mask
    fr = frag_ratio(mask)
    if fr >= FRAGRATIO_THRESHOLDS.get(organ, 0.95):
        return mask
    print(f'  {organ}: FragRatio={fr:.3f} → largest component')
    return largest_component(mask)


# ── correction 2: HU-gated binary dilation for hollow organs ─────────────────

def correct_hollow(mask, ct, organ, claimed=None):
    if mask.sum() == 0 or organ not in HU_WINDOWS:
        return mask
    hu_min, hu_max = HU_WINDOWS[organ]
    struct      = generate_binary_structure(3, 1)
    dilated     = scipy_dilation(mask.astype(bool), structure=struct)
    expansion   = dilated & ~mask.astype(bool)
    not_claimed = ~claimed.astype(bool) if claimed is not None \
                  else np.ones_like(mask, dtype=bool)
    valid     = expansion & (ct >= hu_min) & (ct <= hu_max) & not_claimed
    corrected = (mask.astype(bool) | valid).astype(np.uint8)
    added     = int(valid.sum())
    if added > 0:
        print(f'  {organ}: HU dilation +{added} voxels')
    return corrected


# ── correction 3: gas pocket recovery for colon ───────────────────────────────

def correct_colon(mask, ct, organ, claimed=None, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Gas pocket recovery for colon/intestine (paper Section 3.1):
    1. Find air voxels (-1024 to -350 HU) within 5mm of prediction
    2. Retain only those connected to existing prediction
    3. Expand one step into wall tissue (0 to 80 HU)
    4. Also recover fluid-filled lumen (0 to 30 HU)
    """
    if mask.sum() == 0:
        return mask

    air      = (ct >= HU_AIR_RANGE[0]) & (ct <= HU_AIR_RANGE[1])
    dist_vox = [max(1, int(math.ceil(COLON_GAS_DIST_MM / s))) for s in voxel_spacing]
    struct   = np.ones(dist_vox, dtype=bool)

    # step 1: air voxels within 5mm of prediction
    dilated       = scipy_dilation(mask.astype(bool), structure=struct)
    candidate_air = air & dilated & ~mask.astype(bool)

    # step 2: retain only air connected to existing prediction
    combined     = mask.astype(bool) | candidate_air
    labeled, _   = cc_label(combined)
    colon_labels = set(labeled[mask.astype(bool)].flatten()) - {0}
    conn_air     = np.isin(labeled, list(colon_labels)) & candidate_air
    recovered    = mask.astype(bool) | conn_air

    not_claimed = ~claimed.astype(bool) if claimed is not None \
                  else np.ones_like(mask, dtype=bool)

    # step 3: expand one step into wall tissue (0 to 80 HU)
    struct2  = generate_binary_structure(3, 1)
    expanded = scipy_dilation(recovered, structure=struct2)
    wall_add = expanded & (ct >= 0) & (ct <= 80) & ~recovered & not_claimed

    # step 4: fluid-filled lumen (0 to 30 HU)
    fluid_add = expanded & (ct >= 0) & (ct <= 30) & ~recovered & not_claimed

    corrected = (recovered | wall_add | fluid_add).astype(np.uint8)
    added     = int(corrected.sum()) - int(mask.sum())
    if added > 0:
        print(f'  {organ}: gas recovery +{added} voxels')
    return corrected


# ── correction 4: Dijkstra connectivity for tubular/vascular organs ───────────

def correct_tubular(mask, organ, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Connect fragmented vascular predictions via shortest path (paper Section 3.1).
    Uses exact distance transform to find nearest surface point pairs.
    Removes spurious small components after connecting.
    """
    if mask.sum() == 0:
        return mask
    fr = frag_ratio(mask)
    if fr >= FRAGRATIO_THRESHOLDS.get(organ, 0.85):
        return mask
    labeled, n = cc_label(mask)
    if n <= 1:
        return mask.astype(np.uint8)

    sizes  = [(labeled == i).sum() for i in range(1, n + 1)]
    top_n  = sorted(range(1, n + 1), key=lambda i: sizes[i-1], reverse=True)[:5]
    gap_voxels = DIJKSTRA_GAP_MM.get(organ, 25) / min(voxel_spacing)
    result = mask.copy().astype(np.uint8)

    for i in range(len(top_n)):
        for j in range(i + 1, len(top_n)):
            comp_a = (labeled == top_n[i]).astype(np.uint8)
            comp_b = (labeled == top_n[j]).astype(np.uint8)

            dist_a    = distance_transform_edt(~comp_a.astype(bool))
            dist_b    = distance_transform_edt(~comp_b.astype(bool))

            overlap_a = dist_a * comp_b.astype(float)
            overlap_a[overlap_a == 0] = np.inf
            if np.isinf(overlap_a).all():
                continue
            idx_a = np.unravel_index(np.argmin(overlap_a), overlap_a.shape)

            overlap_b = dist_b * comp_a.astype(float)
            overlap_b[overlap_b == 0] = np.inf
            if np.isinf(overlap_b).all():
                continue
            idx_b = np.unravel_index(np.argmin(overlap_b), overlap_b.shape)

            gap = np.sqrt(sum((a - b)**2 for a, b in zip(idx_a, idx_b)))
            if gap > gap_voxels:
                print(f'  {organ}: fragment gap '
                      f'{gap * min(voxel_spacing):.1f}mm > '
                      f'{DIJKSTRA_GAP_MM.get(organ, 25)}mm, skipping')
                continue

            for t in np.linspace(0, 1, int(gap) + 1):
                pt = tuple(int(round(a + t * (b - a))) for a, b in zip(idx_a, idx_b))
                if all(0 <= p < s for p, s in zip(pt, result.shape)):
                    result[pt] = 1

            print(f'  {organ}: connected fragment '
                  f'({gap * min(voxel_spacing):.1f}mm gap)')

    # remove spurious small components after connecting
    result = remove_noise(result, min_voxels=50)
    return result


# ── main pipeline ─────────────────────────────────────────────────────────────

def atlas_postprocess(pred_labels, ct, label_map, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    ATLAS postprocessing pipeline (paper Section 3.1).
    Args:
        pred_labels:   np.int16 array (D, H, W)
        ct:            np.float32 array, same shape, Hounsfield units
        label_map:     dict mapping organ name → integer label in pred_labels
                       e.g. {'liver': 1, 'spleen': 2, 'colon': 6, ...}
        voxel_spacing: (dx, dy, dz) in mm
    Returns:
        corrected np.int16 array, same shape as pred_labels
    """
    corrected = pred_labels.copy()
    claimed   = (pred_labels > 0).astype(np.uint8)

    # 1. solid organs: largest component (SSM requires PanTS — applied separately)
    print('--- Solid organs ---')
    for organ in SOLID_ORGANS:
        if organ not in label_map:
            continue
        lbl  = label_map[organ]
        mask = (pred_labels == lbl).astype(np.uint8)
        if mask.sum() == 0:
            continue
        fixed = correct_solid(mask, organ, voxel_spacing)
        corrected[corrected == lbl] = 0
        corrected[fixed > 0]        = lbl

    # 2. tubular/vascular: Dijkstra connectivity + noise removal
    print('--- Tubular/vascular ---')
    for organ in TUBULAR_ORGANS:
        if organ not in label_map:
            continue
        lbl  = label_map[organ]
        mask = (pred_labels == lbl).astype(np.uint8)
        if mask.sum() == 0:
            continue
        fixed = correct_tubular(mask, organ, voxel_spacing)
        corrected[corrected == lbl] = 0
        corrected[fixed > 0]        = lbl

    # update claimed after structural corrections
    claimed = (corrected > 0).astype(np.uint8)

    # 3. hollow/small organs: HU-gated binary dilation
    print('--- Hollow organs ---')
    for organ in HOLLOW_ORGANS:
        if organ not in label_map:
            continue
        lbl          = label_map[organ]
        mask         = (pred_labels == lbl).astype(np.uint8)
        if mask.sum() == 0:
            continue
        claimed_excl = ((claimed > 0) & (corrected != lbl)).astype(np.uint8)
        fixed        = correct_hollow(mask, ct, organ, claimed_excl)
        corrected[corrected == lbl] = 0
        corrected[fixed > 0]        = lbl
        claimed = (corrected > 0).astype(np.uint8)

    # 4. colon/intestine: gas pocket recovery
    print('--- Colon/intestine ---')
    for organ in COLON_ORGANS:
        if organ not in label_map:
            continue
        lbl          = label_map[organ]
        mask         = (pred_labels == lbl).astype(np.uint8)
        if mask.sum() == 0:
            continue
        claimed_excl = ((claimed > 0) & (corrected != lbl)).astype(np.uint8)
        fixed        = correct_colon(mask, ct, organ, claimed_excl, voxel_spacing)
        corrected[corrected == lbl] = 0
        corrected[fixed > 0]        = lbl
        claimed = (corrected > 0).astype(np.uint8)

    return corrected.astype(np.int16)


# ── convenience: run from NIfTI files ────────────────────────────────────────

def run_from_files(pred_path, ct_path, output_path, label_map):
    """
    pred_path:   path to prediction NIfTI
    ct_path:     path to CT NIfTI in Hounsfield units
    output_path: path to save corrected NIfTI
    label_map:   dict mapping organ name → integer label
                 e.g. {'liver': 1, 'spleen': 2, 'colon': 6, ...}
    """
    pred_img      = nib.load(pred_path)
    pred_labels   = np.asarray(pred_img.dataobj, dtype=np.int16)
    ct_img        = nib.load(ct_path)
    ct            = np.asarray(ct_img.dataobj, dtype=np.float32)
    voxel_spacing = tuple(float(z) for z in ct_img.header.get_zooms()[:3])

    if ct.shape != pred_labels.shape:
        ct = ndimage.zoom(
            ct,
            tuple(p / c for p, c in zip(pred_labels.shape, ct.shape)),
            order=1
        )

    corrected = atlas_postprocess(pred_labels, ct, label_map, voxel_spacing)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    nib.save(
        nib.Nifti1Image(corrected, pred_img.affine, pred_img.header),
        output_path
    )
    print(f'Saved: {output_path}')
    return corrected


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import glob as _glob

    parser = argparse.ArgumentParser(
        description='ATLAS postprocessing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file:
  python atlas_postprocess.py \\
      -p ./predictions/case.nii.gz \\   # path to prediction
      -c ./ct/case_0000.nii.gz \\       # path to CT in HU
      -o ./corrected/case.nii.gz        # path to save corrected output

  # Folder of predictions (CT folder must contain matching filenames):
  python atlas_postprocess.py \\
      -p ./predictions/ \\              # folder of prediction .nii.gz files
      -c ./ct/ \\                       # folder of CT .nii.gz files (matched by filename stem)
      -o ./corrected/                   # output folder

  # Custom label map:
  python atlas_postprocess.py \\
      -p ./predictions/ -c ./ct/ -o ./corrected/ \\
      --labels liver:12 spleen:17 colon:6 stomach:18
        """
    )
    parser.add_argument('-p', '--pred',   required=True,
                        help='Path to prediction .nii.gz OR folder of prediction files')
    parser.add_argument('-c', '--ct',     required=True,
                        help='Path to CT .nii.gz in HU OR folder of CT files (matched by filename stem)')
    parser.add_argument('-o', '--output', required=True,
                        help='Path to save corrected .nii.gz OR output folder when -p is a folder')
    parser.add_argument('--labels', nargs='*', default=None,
                        metavar='organ:label',
                        help='Custom label map as organ_name:integer pairs. '
                             'If omitted, uses default ATLAS-Net label map.')
    args = parser.parse_args()

    # default ATLAS-Net label map
    DEFAULT_LABEL_MAP = {
        'aorta':               1,
        'adrenal_l':           2,
        'adrenal_r':           3,
        'celiac_artery':       5,   # TUBULAR_ORGANS
        'colon':               6,
        'duodenum':            7,
        'gallbladder':         8,
        'ivc':                 9,
        'kidney_l':            10,
        'kidney_r':            11,
        'liver':               12,
        'pancreas':            13,
        'sma':                 15,  # TUBULAR_ORGANS (superior mesenteric artery)
        'intestine':           16,
        'spleen':              17,
        'stomach':             18,
        'portal_splenic_vein': 19,
        'renal_vein_l':        20,
        'renal_vein_r':        21,
        # 'bladder': None  — not present in this label map
    }

    if args.labels:
        label_map = {}
        for item in args.labels:
            organ, lbl = item.split(':')
            label_map[organ.strip()] = int(lbl.strip())
        print(f'Using custom label map: {label_map}')
    else:
        label_map = DEFAULT_LABEL_MAP
        print(f'Using default ATLAS-Net label map')

    # ── single file mode ──────────────────────────────────────────────────────
    if os.path.isfile(args.pred):
        run_from_files(args.pred, args.ct, args.output, label_map)

    # ── folder mode ───────────────────────────────────────────────────────────
    elif os.path.isdir(args.pred):
        pred_files = sorted(
            _glob.glob(os.path.join(args.pred, '*.nii.gz')) +
            _glob.glob(os.path.join(args.pred, '*.nii'))
        )
        if not pred_files:
            print(f'No .nii.gz files found in {args.pred}')
            exit(1)

        os.makedirs(args.output, exist_ok=True)
        print(f'Found {len(pred_files)} prediction(s)')

        for pred_path in pred_files:
            fname    = os.path.basename(pred_path)
            stem     = fname.replace('.nii.gz', '').replace('.nii', '')

            # match CT by filename stem — tries exact match then _0000 suffix
            ct_path = None
            for suffix in ['.nii.gz', '.nii']:
                for ct_name in [f'{stem}{suffix}', f'{stem}_0000{suffix}']:
                    candidate = os.path.join(args.ct, ct_name)
                    if os.path.exists(candidate):
                        ct_path = candidate
                        break
                if ct_path:
                    break

            if ct_path is None:
                print(f'  ✗ No matching CT found for {fname} in {args.ct} — skipping')
                continue

            out_path = os.path.join(args.output, fname)
            print(f'\n[{stem}]')
            print(f'  pred: {pred_path}')
            print(f'  ct:   {ct_path}')
            print(f'  out:  {out_path}')
            run_from_files(pred_path, ct_path, out_path, label_map)

        print(f'\nDone — processed {len(pred_files)} case(s)')

    else:
        print(f'Error: {args.pred} is not a valid file or folder')
        exit(1)