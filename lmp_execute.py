from LabelMeParser import LabelMap
import optparse
import sys

if __name__ == '__main__':
    parser = optparse.OptionParser("LabelMeParser", version='1.0.0')

    parser.add_option('-j', '--json_path', action='store', dest='json_path', help='Set saved json path')
    parser.add_option('-o', '--origin_path', action='store', dest='origin_path', help='Set saved original image path')
    parser.add_option('-s', '--target_semantic_path', action='store', dest='target_semantic_path', help='Set saved semantic target path')
    parser.add_option('-i', '--target_instance_path', action='store', dest='target_instance_path', help='Set saved instance target path')
    (options, args) = parser.parse_args(sys.argv)

    assert options.json_path is not None, "Please put 'lableme' json file path!"

    # load data map.
    lmp = LabelMap(json_path=options.json_path)

    if options.origin_path is not None:
        # save origin image. --
        lmp.save_original_img(target_path=options.origin_path)

    if options.target_semantic_path is not None:
        # save semantic image.
        lmp.save_semantic_label(target_path=options.target_semantic_path)

    if options.target_instance_path is not None:
        # save instance image.
        lmp.save_instance_label(target_path=options.target_instance_path)
