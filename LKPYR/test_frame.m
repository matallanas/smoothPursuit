 clear;
 close all;
      
%   optical = vision.OpticalFlow('OutputValue', 'Horizontal and vertical components in complex form');

  %figure();
  %set(gcf, 'BackingStore', 'off'); 
  figure();
  
  images(:,:,1) = (imread ('frame10.png')); 
  images(:,:,2) = (imread ('frame11.png')); 
   
%   images(:,:,1) = rgb2gray(imread ('frame10.png')); 
%   images(:,:,2) = rgb2gray(imread ('frame11.png')); 
  
%   images(:,:,1) = (imread ('pic18.bmp')); 
%   images(:,:,2) = (imread ('pic30.bmp')); 
  %%
%%%  images=zeros(36*5,64*5,15);
%   for i = 1:size(images,3)
       
      
      tic                
      % Compute the optical flow for that particular frame.
%       optFlow = step(optical,single(tmp));
      [Vx, Vy,time] = LKPYR(images(:,:,1), images(:,:,2),'numLevels',1,'winSize',21);
      %optFlow_DS = optFlow(r, c);
%       Vy = imag(optFlow);
%       Vx = real(optFlow);  
      %[Vx, Vy] = HornOpticFlow(images, 100.0, 1);
      %%[Vx, Vy] = HornOpticFlowGPU(images, 100.0, 1,k);
      toc
                                                 
      xgrid = 1:5:size(Vx,2);
      ygrid = 1:5:size(Vx,1);
      [xi,yi]=meshgrid(xgrid, ygrid);
      %Vxi = interp2(Vx, xi, yi);
      Vxi = Vx(ygrid,xgrid);
    
      %Vyi = interp2(Vy, xi, yi);           
      Vyi = Vy(ygrid,xgrid);
     
%%%      Vyi=zeros(size(Vxi));
      imshow(images(:,:,1)) 
      hold on;           
      quiver(xgrid, ygrid,100*Vxi, 100*Vyi, 'r');                             
      hold off;
      drawnow;

      save('flow-vector', 'Vx');
%   end
%%%%%%%%%%%%%
