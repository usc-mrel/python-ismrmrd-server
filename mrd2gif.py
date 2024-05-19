#!/usr/bin/python3

import os
import argparse
import h5py
import ismrmrd
import numpy as np
import mrdhelper
from PIL import Image, ImageDraw

def main(args):
    dset = h5py.File(args.filename, 'r')
    if not dset:
        print("Not a valid dataset: %s" % (args.filename))
        return

    dsetNames = dset.keys()
    print("File %s contains %d groups:" % (args.filename, len(dset.keys())))
    print(" ", "\n  ".join(dsetNames))

    if not args.in_group:
        if len(dset.keys()) > 1:
            print("Input group not specified -- selecting most recent")
        args.in_group = list(dset.keys())[-1]

    if args.in_group not in dset:
        print("Could not find group %s" % (args.in_group))
        return

    group = dset.get(args.in_group)
    print("Reading data from group '%s' in file '%s'" % (args.in_group, args.filename))

    # Image data is stored as:
    #   /group/config              text of recon config parameters (optional)
    #   /group/xml                 text of ISMRMRD flexible data header (optional)
    #   /group/image_0/data        array of IsmrmrdImage data
    #   /group/image_0/header      array of ImageHeader
    #   /group/image_0/attributes  text of image MetaAttributes
    isImage = True
    imageNames = group.keys()
    print("Found %d image sub-groups: %s" % (len(imageNames), ", ".join(imageNames)))

    for imageName in imageNames:
        if ((imageName == 'xml') or (imageName == 'config') or (imageName == 'config_file')):
            continue

        image = group[imageName]
        if not (('data' in image) and ('header' in image) and ('attributes' in image)):
            isImage = False

    dset.close()

    if (isImage is False):
        print("File does not contain properly formatted MRD raw or image data")
        return

    dset = ismrmrd.Dataset(args.filename, args.in_group, False)

    groups = dset.list()
    for group in groups:
        if ( (group == 'config') or (group == 'config_file') or (group == 'xml') ):
            continue

        print("Reading images from '/" + args.in_group + "/" + group + "'")

        images = []
        rois   = []
        for imgNum in range(0, dset.number_of_images(group)):
            image = dset.read_image(group, imgNum)

            if ((image.data.shape[0] == 3) and (image.getHead().image_type == 6)):
                # RGB images
                data = np.squeeze(image.data.transpose((2, 3, 0, 1))) # Transpose to [row col rgb]
                data = data.astype(np.uint8)                          # Stored as uint16 as per MRD specification, but uint8 required for PIL
                images.append(Image.fromarray(data, mode='RGB'))
            else:
                for cha in range(image.data.shape[0]):
                    for sli in range(image.data.shape[1]):
                        data = np.squeeze(image.data[cha,sli,...]) # image.data is [cha z y x] -- squeeze to [y x] for [row col]
                        images.append(Image.fromarray(data))

            # Read ROIs
            meta = ismrmrd.Meta.deserialize(image.attribute_string)
            imgRois = []
            for key in meta.keys():
                if not key.startswith('ROI_'):
                    continue

                roi = meta[key]
                x, y, rgb, thickness, style, visibility = mrdhelper.parse_roi(roi)

                if visibility == 0:
                    continue

                imgRois.append((x, y, rgb, thickness))
            rois.append(imgRois)
        print("  Read in %s images" % (len(images)))

        hasRois = any([len(x) > 0 for x in rois])

        # Window/level images
        maxVal = np.median([np.percentile(np.array(img), 95) for img in images])
        minVal = np.median([np.percentile(np.array(img),  5) for img in images])

        imagesWL = []
        for img, roi in zip(images, rois):
            if img.mode != 'RGB':
                if hasRois:
                    # Convert to RGB mode to allow colored ROI overlays
                    data = np.array(img).astype(float)
                    data -= minVal
                    data *= 255/(maxVal - minVal)
                    data = np.clip(data, 0, 255)
                    tmpImg = Image.fromarray(np.repeat(data[:,:,np.newaxis],3,axis=2).astype(np.uint8), mode='RGB')

                    draw = ImageDraw.Draw(tmpImg)
                    for (x, y, rgb, thickness) in roi:
                        draw.line(list(zip(x, y)), fill=(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255), 0), width=int(thickness))
                    imagesWL.append(tmpImg)
                else:
                    data = np.array(img).astype(float)
                    data -= minVal
                    data *= 255/(maxVal - minVal)
                    data = np.clip(data, 0, 255)
                    imagesWL.append(Image.fromarray(data))
            else:
                imagesWL.append(img)

        # Add SequenceDescriptionAdditional to filename, if present
        image = dset.read_image(group, 0)
        meta = ismrmrd.Meta.deserialize(image.attribute_string)
        if 'SequenceDescriptionAdditional' in meta.keys():
            seqDescription = '_' + meta['SequenceDescriptionAdditional']
        else:
            seqDescription = ''

        # Make valid file name 
        gifFileName = os.path.splitext(os.path.basename(args.filename))[0] + '_' + args.in_group + '_' + group + seqDescription + '.gif'
        gifFileName = "".join(c for c in gifFileName if c.isalnum() or c in (' ','.','_')).rstrip()
        gifFileName = gifFileName.replace(" ", "_")
        gifFilePath = os.path.join(os.path.dirname(args.filename), gifFileName)

        print("  Writing image: %s " % (gifFilePath))
        if len(images) > 1:
            imagesWL[0].save(gifFilePath, save_all=True, append_images=imagesWL[1:], loop=0, duration=40)
        else:
            imagesWL[0].save(gifFilePath, save_all=True, append_images=imagesWL[1:])

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MRD image file to animated GIF',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',          help='Input file')
    parser.add_argument('-g', '--in-group',  help='Input data group')

    args = parser.parse_args()

    main(args)
