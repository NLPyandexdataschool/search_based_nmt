import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, dest='big_table_file_name', help='File with big table.')
    parser.add_argument(type=str, dest='target_file_name', help='File name for new table.')
    parser.add_argument(type=str, dest='file_to_search', help='File with words to leave in new table.')
    args = parser.parse_args()
    with open(args.file_to_search) as words_to_search_handler:
        words_to_search = set([line.strip() for line in words_to_search_handler])
    with open(args.big_table_file_name) as big_table_handler,\
         open(args.target_file_name, 'w') as target_handler:
        for line in big_table_handler:
            words = line.strip().split()
            target_handler.write(
                ' '.join([words[0]] + [word for word in words[1:] if word in words_to_search]) + '\n'
            )
