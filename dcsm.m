function dcsm()
run  matlab/vl_setupnn
net =load('../caffe3/caffe-master/models/vgg16_N/pas12/WE20LSE/model/gpconv_aug_100000.mat'); 


%%%%%%%%%%parameters
surid=[11,13,15,18,20,22,25,27,29];
aveimval=[];aveimval(1)=104.00699;aveimval(2)=116.66877;aveimval(3)=122.67892;    
scalerate=[0.8,1,1.2];scalerateid=[5,1,2];%50
DN=20;
bsize=500;
softk=5;
tha=3;%%97

%%%%%%%%%%%preprocessing
im=imread('./img.jpg');
[imh imw imc]=size(im);
starth=(bsize-imh+mod(imh,2))/2+1;
endh=starth+imh-1;
startw=(bsize-imw+mod(imw,2))/2+1;
endw=startw+imw-1;
imbsize=zeros(bsize,bsize,3);
imbsize(starth:endh,startw:endw,:)=single(im);
ecnn={};eim={};ecnnm={};
cid=zeros(1,DN); csid=zeros(1,DN); csval=zeros(1,DN);

dsurfin={};ctar={};
for j3=1:DN
	dsurfin{j3}=zeros(bsize,bsize);
end
sumcnn=zeros(1,DN);
etar={};
%%%%%%%%%%%%%%%recognition
for j3=1:numel(scalerate)
	imrsz=imbsize;
	im_ = single(imrsz) ; % note: 255 range
	rsz=[round(scalerate(j3)*bsize),round(scalerate(j3)*bsize)];
	im_ = imresize(im_, rsz) ;
	aveim=ones(rsz(1),rsz(2),3);
	aveim(:,:,1)=aveimval(1);aveim(:,:,2)=aveimval(2);aveim(:,:,3)=aveimval(3);
	im_ = im_ - aveim;

	%cnn
	cnn = vl_simplenn(net, im_);
	ecnn{j3}=cnn;
	eim{j3}=im_;
	lastcnn2=cnn(38).x;
	[cnnh cnnw cnnc]=size(lastcnn2);
			
	sbsize=[bsize,bsize];
	cnnm=zeros(1,cnnc);
	for cc=1:cnnc
		lastcnn2t=lastcnn2(:,:,cc);
		mval=max(lastcnn2t(:));
		if cnnm(cc) < mval
			cnnm(cc)=mval;
		end
	end
	cnn_tsh=0.5;
	target=find(cnnm > cnn_tsh);
	cid(target)=1;
	sumcnn=sumcnn+cnnm;
	etar{j3}=cnnm > cnn_tsh;
end

target=find(cid==1);

if(numel(target)<4)
	tn=4;
	ptn=tn-numel(target);
	sumcnn(target)=0;
	[sorted sortid]=sort(sumcnn,'descend');
	tid=[target sortid(1:ptn)];
else
	tn=numel(target);
	tid=target;
end
%%%%%%%%%%%%%%%visualization
for j3=1:numel(scalerate)%%% roop for size
	cnn=ecnn{j3};
	lastcnn2=cnn(38).x;
	[cnnh cnnw cnnc]=size(lastcnn2);
	im_=eim{j3};
	et=etar{j3};
	gbps={};
	for t=1:tn
		d=zeros(cnnh,cnnw,cnnc);
		d(:,:,tid(t))=1; %bb image flag
		bp=vl_gbp(net,im_,d,cnn);
		gbps{t}=bp;
	end
	for t=1:tn %%% roop for category
		if et(tid(t)) == 1
			sur_only_sum=zeros(imh,imw);
			dsur_only_sum=zeros(imh,imw);
			gbp1=gbps{t};
			for i1kk=1:numel(surid)%%% roop for layer
				i1=surid(i1kk);
				dsursum=zeros(bsize,bsize);
				for t2=1:tn
					if t==t2
					else
						sur1=abs(gbp1(i1).dzdx);%abs 
						surgbp1=max(sur1,[],3);	
						dsur1=imresize(surgbp1,sbsize,'bilinear');
						gbp2=gbps{t2};
						sur2=abs(gbp2(i1).dzdx);%abs 
						surgbp2=max(sur2,[],3);	
						dsur2=imresize(surgbp2,sbsize,'bilinear');
						dsur=dsur1-dsur2;
						dsur=dsur/max(dsur(:));
						dsur_only_sum=dsur_only_sum+dsur(starth:endh,startw:endw);
						dsur=tanh(tha*dsur);
						if max(dsur(:)) > 0
							dsur(dsur<=0)=1e-20;
							dsurfin{tid(t)}=dsurfin{tid(t)}+dsur;
							dsursum=dsursum+dsur;
						end
					end
				end
			end
		end
	end
end
ttt=find(cid==1);
dir='temp2';
surb1=zeros(imh,imw);
for tt=1:numel(ttt)
	sur=dsurfin{ttt(tt)};
	sur=sur(starth:endh,startw:endw);
	sur=sur/max(sur(:));
	surb1=max(surb1,sur);
	sur=tanh(tha*sur);
	sur(sur<=0)=1e-20;
	sn=sprintf('%s/dcsm_%d.jpg',dir,tt);
	imwrite(sur/max(sur(:)),sn);
end
sur=surb1;
sur=sur/max(sur(:));
sur=tanh(tha*sur);
sn=sprintf('%s/dcsm_%d.jpg',dir,0);
imwrite(sur/max(sur(:)),sn);
