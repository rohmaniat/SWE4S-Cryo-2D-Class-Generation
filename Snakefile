# This Snakefile uses the pre-generated model and sample data to
# predict particle coordinates for three files!

# Get list of input MRC files (without extension)
SAMPLES = glob_wildcards("src/sample_data/{sample}.mrc").sample

rule all:
    input:
        # Output images
        expand("src/visualization/{sample}_image.png", sample=SAMPLES),
        # Ground truth CSV files
        expand("src/visualization/{sample}_table.csv", sample=SAMPLES)

rule predict:
    input:
        mrc="src/sample_data/{sample}.mrc",
        gt="src/sample_data/{sample}.csv"
    output:
        csv="src/visualization/{sample}_table.csv",
        img="src/visualization/{sample}_image.png"
    conda:
        "particle"
    shell:
        """
        python src/predict.py \
            --mrc_file {input.mrc} \
            --output_csv {output.csv} \
            --output_image {output.img} \
            --ground_truth_csv {input.gt} \
            --threshold 0.3
        """