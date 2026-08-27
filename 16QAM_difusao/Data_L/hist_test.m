a = [0.4, 0.5, 0.4, 0.5, 0.6];
% histogram(a,3,'Normalization','pdf')
[counts, centers] = hist(a,3);
counts = counts/sum(counts);
bar(centers, counts, 'hist');
% 
% aes_dif = csvread("comp_enc_vec_dif.csv");
% dif_SPE_dk = csvread("Dif_list_SPE_dk.csv",0,1);
% % 
% [aes_dif_amp, aes_dif_centers] = hist(aes_dif,160);
% [dif_SPE_dk_amp, dif_SPE_dk_centers] = hist(dif_SPE_dk,160);
% aes_dif_amp = aes_dif_amp/sum(aes_dif_amp);
% dif_SPE_dk_amp = dif_SPE_dk_amp/sum(dif_SPE_dk_amp);
% % 
% figure
% subplot(2,1,1)
% bar(aes_dif_centers, aes_dif_amp, 'hist', 'r');
% hold on;
% xlim([0.3,0.7])
% subplot(2,1,2)
% bar(dif_SPE_dk_centers, dif_SPE_dk_amp, 'hist', 'b');
% hold on;
% xlim([0.3,0.7])