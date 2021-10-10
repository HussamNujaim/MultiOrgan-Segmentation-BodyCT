import numpy as np
import SimpleITK as sitk
# reference: from DLTK
def resample_img(itk_image, out_spacing=[2, 2, 2], is_label=False):
    # resample images to 2mm spacing with simple itk

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.floor(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.floor(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.floor(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
        #resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(itk_image)
