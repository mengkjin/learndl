def combine_htmls(html_paths_, des_path_):
    with open(des_path_, "wb") as f:
        for src in html_paths_:
            f.write(open(src, "rb").read())