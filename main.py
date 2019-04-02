import os
import process
import config
import argparse
from DM.file_manage import RotatedNiiFileManager, LabelNiiFileManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['calc', 'seg', 'both'])
    parser.add_argument('-i', '--image', required=True)
    parser.add_argument('-l', '--label', required=True)
    parser.add_argument('-o', '--output')
    parser.add_argument('-d', '--data')
    args = parser.parse_args()
    assert os.path.isfile(args.label)
    assert os.path.isfile(args.image)
    image_nii = RotatedNiiFileManager(args.image)
    image_nii.load()
    image_nii.normalize()

    label_nii = LabelNiiFileManager(args.label)
    label_nii.load()
    show = args.mode != 'calc'
    quad_seg_point, mid_num, pons_area, midbrain_area = process.first_stage.run(
        image_nii, label_nii
    )
    mcp_mean_width, mcp_show_info = process.second_stage.run(
        image_nii, label_nii, quad_seg_point, mid_num, show
    )
    scp_mean_width, scp_show_info = process.third_stage.run(
        image_nii, label_nii, quad_seg_point, mid_num, show
    )
    if args.mode == 'calc':
        process.output.calc(pons_area, midbrain_area, mcp_mean_width, scp_mean_width, file_name=args.data)
    elif args.mode == 'seg':
        assert args.output
        process.output.save_to_sitk(
            args.output, image_nii.size, mid_num,
            quad_seg_point, mcp_show_info, scp_show_info
        )
    elif args.mode == 'both':
        assert args.output
        process.output.save_to_sitk(
            args.output, image_nii.size, mid_num,
            quad_seg_point, mcp_show_info, scp_show_info
        )
        process.output.calc(pons_area, midbrain_area, mcp_mean_width, scp_mean_width, file_name=args.data)
    else:
        raise TypeError('mode must be one of {calc, seg, both}')
