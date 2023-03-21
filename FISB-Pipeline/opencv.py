from lib import *
import metrics

if __name__ == '__main__':
    # Read images
    
    dir = sys.argv[1]
    files = os.listdir(dir)
    images = []
    for i in range(len(files)):
        img = cv2.imread(os.path.join(dir, files[i]))
        assert img is not None, "Image not found"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    # Stitch images
    stitcher = cv2.Stitcher.create()
    result, pano = stitcher.stitch(images)

    # Write result
    folder = dir.split('/')[-1]
    folder = folder.split('.')[-1]
    pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join("output/", folder + ".png"), pano)
    orgImg = cv2.imread(os.path.join("fisb_dataset/super/" + folder + ".jpeg"))
    assert orgImg is not None, "Image not found"
    metrics.showStatistics(pano,orgImg)