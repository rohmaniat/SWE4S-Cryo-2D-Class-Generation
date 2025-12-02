# This Snakefile uses the pre-generated model and sample data to
# predict particle coordinates for three files!

# Get list of input MRC files (without extension)
SAMPLES = glob_wildcards("src/sample_data/{sample}.mrc").sample

rule all:
    input:
        # Output images
        expand("results/{sample}_output.png", sample=SAMPLES),
        # Ground truth CSV files
        expand("results/{sample}_ground_truth.csv", sample=SAMPLES)

rule predict:
    input:
        mrc="src/sample_data/{sample}.mrc"
    output:
        csv="visualization/{sample}.csv",
        img="visualization/{sample}_output.png",
        gt="visualization/{sample}_ground_truth.csv"
    conda:
        "particle"
    shell:
        """
        python predict.py \
            --mrc_file {input.mrc} \
            --output_csv {output.csv} \
            --output_image true \
            --ground_truth_csv true
        """