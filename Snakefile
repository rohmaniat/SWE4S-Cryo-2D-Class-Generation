rule all:
    input: output_list

rule download_data:
    output: gdp = "src/IMF_GDP.csv",
            co2 = "src/Agrofood_co2_emission.csv"
    shell:
        """
        curl -L "https://docs.google.com/uc?export=download&id=1Wytf3ryf9EtOwaloms8HEzLG0yjtRqxr" -o {output.co2}
        curl -L "https://docs.google.com/uc?export=download&id=1tuoQ9UTW_XRKgBOBaTLtGXh8h0ytKvFp" -o {output.gdp}
        """

rule find_data:
    input: gdp = "IMF_GDP.csv",
           co2 = "Agrofood_co2_emission.csv"
    output: "src/GDP_CO2_output_{country}.csv"
    shell:
        """
        python3 src/fire_gdp.py \
            --gdp_file {input.gdp} \
                --co2_file {input.co2} \
                    --country {wildcards.country}
        """

rule scatter_plot:
    input: "src/GDP_CO2_output_{country}.csv"
    output: "src/scatter_outputs/scatterplot_{country}.png"
    shell:
        """
        python3 src/scatter.py \
            --data_file {input} \
                --out_file {output} \
                    --title "GDP vs CO2 emissions" \
                        --x_axis "annual GDP" \
                            --y_axis "annual CO2 emission"
        """