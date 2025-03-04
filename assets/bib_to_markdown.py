import bibtexparser

with open('blogpost_bib.bib', 'r', encoding='utf-8') as f:
    bib_content = f.read()

bib_database = bibtexparser.loads(bib_content)


def order_names(author):
    if "," in author:
        last_name, first_name = author.split(',')
        author = f"{first_name} {last_name}"

    author = author.strip(" ")
    return author

for entry in bib_database.entries:
    authors = entry['author'].split(' and ')
    first_author = authors[0]
    first_author = order_names(first_author)  

    if len(authors) > 2:
        first_author = first_author + ' et al.'
    elif len(authors) == 2:
        first_author = first_author + ' and ' + order_names(authors[1])
    else:
        first_author = first_author

    print(f"[_{entry['title'].strip(' ')}_]({entry['url']}), {first_author}, {entry['year']}")
    print('\n')

