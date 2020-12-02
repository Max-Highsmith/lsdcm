import subprocess

for chro in [4,16,14,20]:
    networks = ['hicsr','down', 'hicplus', 'deephic', 'vehicle']
    for cur_net in networks:
        hic_metric_samples = open("hicqc_inputs/metric_"+cur_net+"_"+str(chro)+".samples", 'w')
        hic_metric_pairs   = open("hicqc_inputs/metric_"+cur_net+"_"+str(chro)+".pairs", 'w')
        PAIR_STRING        = "Original_"+str(chro)+"\t"+cur_net+"_"+str(chro)+"\n"
        SAMPLE_STRING      = "Original_"+str(chro)+"\t"\
                "/home/heracles/Documents/Professional/Research/lsdcm/hicqc_inputs/original_"+str(chro)+".gz\n"\
                ""+cur_net+"_"+str(chro)+"\t"\
                "/home/heracles/Documents/Professional/Research/lsdcm/hicqc_inputs/"+cur_net+"_"+str(chro)+".gz\n"
        hic_metric_samples.write(SAMPLE_STRING)
        hic_metric_pairs.write(PAIR_STRING)
        hic_metric_samples.close()
        hic_metric_pairs.close()



