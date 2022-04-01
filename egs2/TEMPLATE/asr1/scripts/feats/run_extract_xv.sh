#Training set
to_extracted_data=lwd/tr_no_dev
stage=1
endstage=2
if [[ $stage -le 1 && 1 -le $endstage ]];then
	### Start

	# Fixed dir and make sure that the various files in a data directory are correctly sorted and filtered
	for data in $to_extracted_data;do
		subtools/kaldi/utils/fix_data_dir.sh data/$data
	done

	# Make features for data
	for data in $to_extracted_data;do
		subtools/makeFeatures.sh data/$data fbank subtools/conf/sre-fbank-81.conf
	done

	# Compute VAD for data
	for data in $to_extracted_data;do
		subtools/computeVad.sh data/$data subtools/conf/vad-5.0.conf
	done
		
	# Get the copies of dataset which is labeled by a prefix like fbank_81 or mfcc_23_pitch etc.
	for data in $to_extracted_data;do
		subtools/newCopyData.sh fbank_81 $data 
	done
fi

if [[ $stage -le 2 && 2 -le $endstage ]];then
	## Pytorch x-vector extracting
	
	python3 pyscripts/feats/extractingXvector.py

fi