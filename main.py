from argparse import ArgumentParser

from models.CGAN.gen_run_script import generate
from models.CGAN.training_script import training


def parse_fade_ins(fade_in: str):
    # Split the string by '-' and convert each part to an integer
    fade_ins = fade_in.split('-')
    # If there's only one number, repeat it 6 times
    if len(fade_ins) == 1:
        fade_ins = [int(fade_ins[0])] * 6
    else:
        fade_ins = [int(fade) for fade in fade_ins]
    return fade_ins


def parse_epochs(epochs: str):
    # Split the string by '-' and convert each part to an integer
    epochs_list = epochs.split('-')
    # Ensure the list has exactly 6 numbers
    if len(epochs_list) != 6:
        raise ValueError("Epochs string must contain exactly 6 numbers.")
    epochs_list = [int(epoch) for epoch in epochs_list]
    return epochs_list


def main():
    parser = ArgumentParser(description="Zelda SNES map generation using various GANs")
    parser.add_argument(
        '-model',
        choices=['pix2pix', 'progan', 'sngan'],
        help='Specify the model (pix2pix, progan, sngan)',
        required=True
    )

    parser.add_argument(
        '-method',
        choices=['train', 'test', 'generate'],
        help='Specify the method (train, test, generate)',
        required=True
    )

    parser.add_argument(
        '-i', '--input',
        help='Specify the input directory',
        required=True
    )
    parser.add_argument(
        '-o', '--output',
        help='Specify the output directory',
        required=True
    )

    parser.add_argument(
        '-w', '--weights',
        help='Specify the folder containing weight files',
        required=False
    )

    parser.add_argument(
        "-f", "--fade_ins",
        help="Input 6 numbers separated with '-' or 1 number between 1 and 100",
        required=False
    )

    parser.add_argument(
        "-ep", "--epochs",
        help="Input 6 numbers separated with '-', "
             "each represents a number of epochs a resolution will get trained with",
        required=False
    )

    parser.add_argument(
        "-np", "--num_pics",
        help="Enter a number of the pictures you want to generate",
        required=False
    )

    args = parser.parse_args()
    
    print(f"Using model: {args.model}")
    print(f"Using method: {args.method}")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    if args.weights:
        print(f"Weights directory: {args.weights}")
    if args.fade_ins:
        print(f"Fade ins in %: {parse_fade_ins(args.fade_ins)}")
    if args.epochs:
        print(f"Epochs per resolution (1-6): {parse_epochs(args.epochs)}")
    if args.num_pics:
        print(f"Number of pictures that will be generated: {args.num_pics}")

    if args.model == "progan":
        if args.method == "train":
            training(parse_epochs(args.epochs), parse_fade_ins(args.fade_ins), args.output, args.input)
        if args.method == "generate":
            generate(args.num_pics, args.input, args.output)
    elif args.model == "sngan":
        if args.method == "train":
            print("bin da")
        if args.method == "generate":
            print("bin da")
        if args.method == "test":
            print("bin da")
    elif args.model == "pix2pix":
        if args.method == "train":
            print("bin da")
        if args.method == "generate":
            print("bin da")
        if args.method == "test":
            print("bin da")


if __name__ == "__main__":
    main()
