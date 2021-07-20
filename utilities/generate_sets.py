import argparse
import random

import struct



def parse_args():
    parser = argparse.ArgumentParser(
        description="Set-Generator for the subset sum problem"
    )

    parser.add_argument(
        "--number_of_elements", "-n", type=int, default=100,
        help="Number of elements in the set"
    )

    parser.add_argument(
        "--negative", action="store_true",
        help="Set this flag to include negative numbers"
    )

    parser.add_argument(
        "--floating_numbers", "--floats", action="store_true",
        help="Set this flag to include floating numbers (this excludes negative and positive infinity)"
    )

    parser.add_argument(
        "--num_bytes", "-b", type=int, choices=[4, 8], default=4,
        help="Set the number of bytes for"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.num_bytes == 4:
        struct_i = "i"
        struct_f = "f"
    elif args.num_bytes == 8:
        struct_i = "l"
        struct_f = "d"



    final_set = set()

    while len(final_set) != args.number_of_elements:

        b = random.randbytes(args.num_bytes)

        if args.floating_numbers:
            next_element = struct.unpack(struct_f, b)[0]
            if next_element == float("inf") or next_element == float("-inf"):
                continue
        else:
            next_element = struct.unpack(struct_i, b)[0]

        if not args.negative and next_element < 0:
            continue

        final_set.add(next_element)

    print(final_set)