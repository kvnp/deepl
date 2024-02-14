from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="Zelda SNES map generation using various GANs")
    parser.add_argument(
        'model',
        choices=['pix2pix', 'progan', 'sngan'],
        help='Specify the model (pix2pix, progan, sngan)',
        required=True
    )

    parser.add_argument(
        'method',
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

    args = parser.parse_args()
    
    print(f"Using model: {args.model}")
    print(f"Using method: {args.method}")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    if args.weights:
        print(f"Weights directory: {args.weights}")

if __name__ == "__main__":
    main()
