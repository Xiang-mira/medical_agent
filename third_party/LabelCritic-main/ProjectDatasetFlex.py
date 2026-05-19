import os
import argparse
import ast
import importlib
import json
try:
    import AnnotationVLM.Projection as pj
except:
    import Projection as pj
importlib.reload(pj)

def parse_organs_arg(organs_str):
    # Check if the string starts with '[' and ends with ']'
    if organs_str.startswith('[') and organs_str.endswith(']'):
        # Remove the square brackets and split the string by commas
        organs_list = organs_str[1:-1].split(',')
        # Strip any extra spaces from the organ names
        organs_list = [organ.strip() for organ in organs_list]
        return organs_list
    # Return as a single element list if it's not a list format
    return None


def main():
    parser = argparse.ArgumentParser(description='Process paths for data projection and composition.')

    # Arguments with defaults
    parser.add_argument('--good_folder', default='/mnt/sdc/pedro/ErrorDetection/revised_cropped/',
                        help='Path to the good samples directory (revised_cropped_projection).')
    parser.add_argument('--bad_folder', default='/mnt/sdc/pedro/ErrorDetection/cropped_nnunet_results_250Epch/',
                        help='Path to the first bad samples directory.')
    parser.add_argument('--output_dir1', default='compose_nnUnet_ProGT',
                        help='Output directory for the first composite dataset.')
    parser.add_argument('--good_folder_mask', default=None, help='Path to masks for good folder.')
    parser.add_argument('--bad_folder_mask', default=None, help='Path to masks for bad folder.')
    parser.add_argument('--organ', default='none')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_processes', default='10')
    parser.add_argument('--file_list', default=None)
    parser.add_argument('--restart', action='store_true',default=False)
    parser.add_argument('--no_composite_images', action='store_true',default=False)
    parser.add_argument('--axis', default=1,type=int)


    args = parser.parse_args()


    good_folder = args.good_folder
    bad_folder = args.bad_folder
    output_dir1 = args.output_dir1

    if args.good_folder_mask is not None:
        good_folder_mask = args.good_folder_mask
    else:
        good_folder_mask = good_folder
    if args.bad_folder_mask is not None:
        bad_folder_mask = args.bad_folder_mask
    else:
        bad_folder_mask = bad_folder
    #print(os.listdir(bad_folder))
    
    organs=parse_organs_arg(args.organ)
    if organs is None:
        if args.organ=='kidneys':
            organs=['kidney_left','kidney_right']
        elif '00' in os.listdir(bad_folder_mask)[0]:
            if args.organ=='none':
                if 'segmentations' in os.listdir(bad_folder_mask+'/'+os.listdir(bad_folder_mask)[0]):
                    organs=[item[:item.rfind('.nii.gz')] for item in os.listdir(bad_folder_mask+'/'+os.listdir(bad_folder_mask)[0]+'/segmentations')]
                elif 'predictions' in os.listdir(bad_folder_mask+'/'+os.listdir(bad_folder_mask)[0]):
                    organs=[item[:item.rfind('.nii.gz')] for item in os.listdir(bad_folder_mask+'/'+os.listdir(bad_folder_mask)[0]+'/predictions')]
            else:
                organs=[args.organ]
        else:
            organs=os.listdir(bad_folder_mask)

    print('Organs:',organs)

    if args.file_list is not None:
        with open(args.file_list, 'r') as file:
            file_list_loaded = json.load(file)
        

    file_list={}
    for organ in organs:
        if '00' in os.listdir(bad_folder_mask)[0]:
            #file_list[organ]=os.listdir(bad_folder_mask)
            pth=bad_folder_mask
        else:
            #file_list[organ]=os.listdir(os.path.join(bad_folder_mask,organ))
            pth=os.path.join(bad_folder_mask,organ)

        file_list[organ]=[f for f in os.listdir(pth) if (os.path.isdir(os.path.join(pth, f,'segmentations')))]
        file_list[organ]=[f for f in file_list[organ] \
                     if (os.path.isfile(os.path.join(pth, f,'segmentations',organ+'.nii.gz')) or \
                         os.path.isfile(os.path.join(pth, f,'predictions',organ+'.nii.gz')))]
        
        if '00' in os.listdir(bad_folder_mask)[0]:
            #file_list[organ]=os.listdir(bad_folder_mask)
            pth=good_folder_mask
        else:
            #file_list[organ]=os.listdir(os.path.join(bad_folder_mask,organ))
            pth=os.path.join(good_folder_mask,organ)

        file_list[organ]=[f for f in file_list[organ] if (os.path.isdir(os.path.join(pth, f,'segmentations')))]
        file_list[organ]=[f for f in file_list[organ] \
                     if (os.path.isfile(os.path.join(pth, f,'segmentations',organ+'.nii.gz')) or \
                         os.path.isfile(os.path.join(pth, f,'predictions',organ+'.nii.gz')))]
        #print(organ,len(file_list[organ]))

        

    #get intersection between file list and file_list_loaded
    if args.file_list is not None:
        for organ in organs:
            file_list[organ]=list(set(file_list[organ])&set(file_list_loaded[organ]))
            
    for organ in organs:
        if 'right' in organ:
            file_list[organ]=list(set(file_list[organ]+file_list[organ.replace('right','left')]))
        if 'left' in organ:
            file_list[organ]=list(set(file_list[organ]+file_list[organ.replace('left','right')]))
    


    # Define projection paths
    good_projection_path = good_folder_mask.rstrip('/') + '_projection'
    bad_projection_path = bad_folder_mask.rstrip('/') + '_projection'

    print(file_list)

    # Ensure projection directories exist
    try:
        os.makedirs(good_projection_path, exist_ok=True)
    except:
        good_projection_path ='./'+good_projection_path[good_projection_path.rfind('/')+1:]
        os.makedirs(good_projection_path, exist_ok=True)
    try:
        os.makedirs(bad_projection_path, exist_ok=True)
    except:
        bad_projection_path ='./'+bad_projection_path[bad_projection_path.rfind('/')+1:]
        os.makedirs(bad_projection_path, exist_ok=True)
        

    # Project files
    for organ in organs:
        if 'all_classes' in organ:
            continue
        #print(file_list[organ])
        if organ != 'none':
            if '00' not in os.listdir(bad_folder_mask)[0]:
                src_ct = os.path.join(good_folder, organ)
                src_mask = os.path.join(good_folder_mask, organ)
            else:
                src_ct = good_folder
                src_mask = good_folder_mask

            destination = os.path.join(good_projection_path, organ)
        else:
            src = good_folder
            destination = good_projection_path
        print(src_mask, destination)
        pj.project_files(
            ct_pth=src_ct,
            mask_pth=src_mask,
            destin=destination,
            file_list=file_list[organ],
            organ=organ,
            device=args.device,
            num_processes=int(args.num_processes),
            skip_existing=(not args.restart),
            axis=args.axis
        )
        if organ != 'none':
            if '00' not in os.listdir(bad_folder_mask)[0]:
                src_ct = os.path.join(bad_folder, organ)
                src_mask = os.path.join(bad_folder_mask, organ)
            else:
                src_ct = bad_folder
                src_mask = bad_folder_mask
            destination = os.path.join(bad_projection_path, organ)
        else:
            src = bad_folder
            destination = bad_projection_path
        pj.project_files(
            ct_pth=src_ct,
            mask_pth=src_mask,
            destin=destination,
            file_list=file_list[organ],
            organ=organ,
            device=args.device,
            num_processes=int(args.num_processes),
            skip_existing=(not args.restart),
            axis=args.axis
        )
        
        # Composite datasets
        pj.composite_dataset(
            output_dir=output_dir1,
            good_path=good_projection_path,
            bad_path=bad_projection_path,
            organ=organ,
            fast= args.no_composite_images,
            file_list=file_list,
            axis=args.axis
        )

    # Join left and right datasets
    for organ in organs:
        if 'left' in organ:
            pj.join_left_and_right_dataset(
                os.path.join(output_dir1, organ),
                os.path.join(output_dir1, organ.replace('left', 'right')),
                os.path.join(output_dir1, organ.replace('_left', 's'))
            )

if __name__ == '__main__':
    main()