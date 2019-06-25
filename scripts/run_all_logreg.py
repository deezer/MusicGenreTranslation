import os
from subprocess import Popen
from tag_translation.conf import FRAC_MAX

EPOCHS=500
LEARNING_RATE=0.5
BATCH_SIZE=100000
EVAL_EVERY=EPOCHS
SIGMA=-1


def format_command_map(script, out_dir, translation_table, frac, sources, target, bias_reg):
    out = ["python", script, "-o", out_dir, "-f", str(frac), "--eval-every", str(EVAL_EVERY), "--epochs", str(EPOCHS),
            "--lr", str(LEARNING_RATE), "--tr-table", translation_table, "--sources"] + sources + \
           ["--target", target, "--batch-size", str(BATCH_SIZE)]
    if bias_reg is not None:
        out.extend([ "--bias-reg", str(bias_reg)])
    return out


def format_command_ml(script, out_dir, frac, sources, target):
    return ["python", script, "-o", out_dir, "-f", str(frac),
                   "--sources"] + sources +  ["--target", target]


def format_command(script, out_dir, translation_table, frac, sources, target, bias_reg):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
    if script == "train_map_logreg.py":
        return format_command_map(script_path, out_dir, translation_table, frac, sources, target, bias_reg)
    return format_command_ml(script_path, out_dir, frac, sources, target)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    import  argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", required=True)
    parser.add_argument("--tr-table", required=False, default=None,
                        help="If passed, this is the pass of the translation"
                             " table. This will be used to define the prior "
                             "distribution when training the MAP objective.")
    parser.add_argument("--bias-reg", type=float, required=False, default=None,
                        help="Bias regularization parameter")
    parser.add_argument("-o", "--out", required=True, help="Where to write the results.")
    args = parser.parse_args()
    sources = list({"lastfm", "tagtraum", "discogs"} - {args.target})
    script = "ml_logreg.py"
    if args.tr_table is not None:
        script = "train_map_logreg.py"

    for frac in range(-13, -FRAC_MAX):
        frac_dir = os.path.join(args.out, "results_frac_{}".format(-frac))
        mkdir(frac_dir)
        command = format_command(script, frac_dir, args.tr_table, frac, sources, args.target, args.bias_reg)
        print("Running command {}".format(" ".join(command)))
        with open(os.path.join(frac_dir, "log"), "w") as logfile:
            p = Popen(command, stdout=logfile, stderr=logfile)
            p.communicate()
            assert p.returncode == 0
