from pathlib import Path

p = Path("test.py")
text = p.read_text(encoding="utf-8")

old1 = """    result_path_input = result_path+'/input/'
    os.makedirs(result_path_gt, exist_ok=True)
    os.makedirs(result_path_out, exist_ok=True)
    os.makedirs(result_path_input, exist_ok=True)"""

new1 = """    result_path_input = result_path+'/input/'
    result_path_raw = result_path+'/raw/'
    os.makedirs(result_path_gt, exist_ok=True)
    os.makedirs(result_path_out, exist_ok=True)
    os.makedirs(result_path_input, exist_ok=True)
    os.makedirs(result_path_raw, exist_ok=True)"""

if old1 not in text:
    raise RuntimeError("Could not find result-folder block. File left unchanged.")

text = text.replace(old1, new1, 1)

old2 = """        # Similar to LLFlow, we follow a similar way of 'Kind' to finetune the overall brightness"""

new2 = """        # Save RAW ReDDiT prediction before GT-mean brightness adjustment.
        util.save_img(
            normal_img, '{}/{}_normal_raw.png'.format(result_path_raw, idx))

        # Similar to LLFlow, we follow a similar way of 'Kind' to finetune the overall brightness"""

if old2 not in text:
    raise RuntimeError("Could not find GT-adjustment block. File left unchanged.")

text = text.replace(old2, new2, 1)

p.write_text(text, encoding="utf-8")

print("PATCH SUCCESSFUL")
print("Raw predictions will be saved in results/raw/")
