# indexing bam file
for filename in ./example_data/*.bam; do
        echo $filename
        samtools index $filename 

done

# create a folder for read depth data of the samples
mkdir read_depths

# generating the read depth data of the samples
for filename in ./example_data/*.bam; do
    basename $filename
    f="$(basename -- $filename)"
    sambamba depth base -L hglft_genome_64dc_dcbaa0.bed $filename > ./read_depths/"$f.txt"

done

mkdir processed_samples

# run the preprocess script preprocess_sample.py
python ./scripts/preprocess_sample.py --readdepth ./read_depths --output ./processed_samples --target hglft_genome_64dc_dcbaa0.bed