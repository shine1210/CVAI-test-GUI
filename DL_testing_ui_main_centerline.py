from PyQt5 import QtCore,QtWidgets
from PyQt5.QtGui import QImage,QMouseEvent
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox,QInputDialog,QLineEdit
from os import execl,listdir

from pydicom import dcmread
from scipy.io import loadmat,savemat
from h5py import File

from tkinter.filedialog import askdirectory,askopenfilename,asksaveasfilename
from tkinter import Tk

from cv2 import cvtColor,COLOR_BGR2RGB,EVENT_MOUSEMOVE,EVENT_FLAG_RBUTTON,namedWindow,WINDOW_NORMAL,WINDOW_FREERATIO,imshow,selectROI,waitKey,destroyAllWindows

import DL_testing_ui_v1,sys
import numpy as np
import vtk
from skimage.morphology import skeletonize_3d

from model_nofrangi import unet as unet_NF
from models_relu import unet as unet_F

from keras import backend as K

from numpy import sqrt

class VnetWindow(DL_testing_ui_v1.Ui_MainWindow):

	def __init__(self,MainWindow):
		super().setupUi(MainWindow)

		# self.state.setText('Welcome To AI Diagnosis Tool')

		self.LOAD_DATA.clicked.connect(self.load_data)
		self.TEST.clicked.connect(self.testing)
		self.VESSELRENDER.clicked.connect(self.render)
		self.RENDER_WITH_G.clicked.connect(self.render_with_ground)
		self.SAVE_RESULT.clicked.connect(self.save_result)

		self.image_slider.valueChanged.connect(self.ch_slice)

		self.CLOSE.clicked.connect(self.close)
		self.RESET.clicked.connect(self.restart_program)

		self.model_F=unet_F()
		self.model_NF=unet_NF()
		# self.model.load_weights('D:/unet_loaddata/test_ui/single_gpu_unet.h5')
	
	def input_type(self):
		if self.TDM_INPUT.isChecked()==True:
			return 'TYPE_TDM'
		elif self.DICOM_INPUT.isChecked()==True:
			return 'TYPE_DICOM'
		elif self.NPY_INPUT.isChecked()==True:
			pass

	def frangi_type(self):
		if self.FRANGI.isChecked()==True:
			return 'USE_FRANGI'
		elif self.NO_FRANGI.isChecked()==True:
			return 'NO_FRANGI'

	def getText(self,matdata):
		inputBox=QInputDialog()
		inputBox.setInputMode(0)
		inputBox.setWindowTitle('MatFileKeyInputDialog')
		itemlist=list()
		for key in matdata.keys():
			itemlist.append(key)
		inputBox.setComboBoxItems(itemlist)
		inputBox.setComboBoxEditable(False)
		inputBox.setLabelText('Please Input MatFile Key')
		inputBox.setOkButtonText(u'Ok')
		inputBox.setCancelButtonText(u'Cancel')
		if inputBox.exec_() and inputBox.textValue()!='':
			return inputBox.textValue()

	def NormlizDcm(self,dicom_set,top=600,bot=-200):

		m,n=dicom_set.shape
		dcm_float=dicom_set.astype(np.float)
		dcm_uint8=np.zeros((m,n,3),np.uint8)

		dcm_float[dcm_float>top]=top
		dcm_float[dcm_float<bot]=bot
		dcm_float[:,:]=255*((dcm_float[:,:]-dcm_float.min())/(dcm_float.max()-dcm_float.min()))
		dcm_uint8[:,:,0]=dcm_float[:,:]
		dcm_uint8[:,:,1]=dcm_float[:,:]
		dcm_uint8[:,:,2]=dcm_float[:,:]

		return dcm_uint8


	def Dice(self,g,p):
		g=g>0
		p=p>0
		smooth = 1e-5
		# g_f = g.reshape(-1)
		# p_f = p.reshape(-1)
		intersection = g * p
		return (2* (intersection.sum())) / (g.sum() + p.sum())

	def load_data(self):
		self.state.setText('Loading Data')

		global dicom_list_array,cx,cy,cm,cn,dicom_list_array_crop
		dicom_list=list()
		inputType=self.input_type()
		Ftype=self.frangi_type()

		if inputType=='TYPE_DICOM':

			root=Tk()
			root.withdraw()
			fille_path=askdirectory()

			try:
				file_list=listdir(fille_path)
			except FileNotFoundError:
				pass
			else:
				file_list=listdir(fille_path)

				for name in file_list:
					dicom_path=fille_path+'/'+name
					try:
						dicom_image=dcmread(dicom_path).pixel_array
					except AttributeError:
						pass
					else:
						dicom_list.append(np.transpose(dcmread(dicom_path).pixel_array))
				dicom_list_array=np.transpose(np.array(dicom_list))
				
				cx,cy,cm,cn=self.dicom_Crop(dicom_list_array)
				dicom_list_array_crop=dicom_list_array[cy:cy+cm,cx:cx+cn,:]
				print(dicom_list_array_crop.shape)
				self.image_slider.setMinimum(0)
				self.image_slider.setMaximum(dicom_list_array.shape[2]-1)
				self.state.setText('Done')

		elif inputType=='TYPE_TDM':
			# global dcm_nor
			global vessel_list_array_crop
			root=Tk()
			root.withdraw()
			fille_path=askopenfilename(filetypes = (("mat files","*.mat"),("all files","*.*")))
			try:
				TDM_data=loadmat(fille_path)
			except FileNotFoundError:
				flag=False
			except ValueError:
				self.ErrorMsg()
				flag=False
			except NotImplementedError:
				flag='v73'
			else:
				flag=True

			if Ftype=='NO_FRANGI':
				
				if flag==False:
					pass

				elif flag=='v73':
					TDM_data=File(fille_path)
					TDMKey=self.getText(TDM_data)
					VesselKey=self.getText(TDM_data) #
					try:
						dicom_list_array=TDM_data[TDMKey]
					except KeyError:
						self.ErrorMsg(input='No Tag Name TDM')
						flag=False
					if flag==False:
						pass
					else:
						TDM_image_v=TDM_data[TDMKey]
						dicom_list_array=np.transpose(TDM_image_v[()])
						# dcm_nor=self.NormlizDcm(dicom_list_array)

				else:
					TDM_data=loadmat(fille_path)
					TDMKey=self.getText(TDM_data)
					VesselKey=self.getText(TDM_data) #
					try:
						dicom_list_array=TDM_data[TDMKey]
					except KeyError:
						self.ErrorMsg(input='No Tag Name TDM')
						flag=False
					if flag==False:
						pass
					else:
						dicom_list_array=TDM_data[TDMKey]
					vessel_list_array=TDM_data[VesselKey] #
						# dcm_nor=self.NormlizDcm(dicom_list_array)
				try:
					dicom_list_array
				except NameError:
					pass
				else:
					cx,cy,cm,cn=self.dicom_Crop(dicom_list_array)
					dicom_list_array_crop=dicom_list_array[cy:cy+cm,cx:cx+cn,:]
					vessel_list_array_crop=vessel_list_array[cy:cy+cm,cx:cx+cn,:] #
					print(dicom_list_array_crop.shape)
					self.image_slider.setMinimum(0)
					self.image_slider.setMaximum(dicom_list_array.shape[2]-1)
					self.state.setText('Done')

			elif Ftype=='USE_FRANGI':
				global frangi_list_array_crop
				if flag==False:
					pass

				elif flag=='v73':
					TDM_data=File(fille_path)
					TDMKey=self.getText(TDM_data)
					FrangiKey=self.getText(TDM_data)
					VesselKey=self.getText(TDM_data) #
					try:
						dicom_list_array=TDM_data[TDMKey]
					except KeyError:
						self.ErrorMsg(input='No Tag Name TDM')
						flag=False
					if flag==False:
						pass
					else:
						TDM_image_v=TDM_data[TDMKey]
						dicom_list_array=np.transpose(TDM_image_v[()])
						# dcm_nor=self.NormlizDcm(dicom_list_array)

				else:
					TDM_data=loadmat(fille_path)
					TDMKey=self.getText(TDM_data)
					FrangiKey=self.getText(TDM_data)
					VesselKey=self.getText(TDM_data) #
					try:
						dicom_list_array=TDM_data[TDMKey]
					except KeyError:
						self.ErrorMsg(input='No Tag Name TDM')
						flag=False
					if flag==False:
						pass
					else:
						dicom_list_array=TDM_data[TDMKey]
						frangi_list_array=TDM_data[FrangiKey]
						vessel_list_array=TDM_data[VesselKey] #
						# dcm_nor=self.NormlizDcm(dicom_list_array)
				try:
					dicom_list_array
				except NameError:
					pass
				else:
					cx,cy,cm,cn=self.dicom_Crop(dicom_list_array)
					dicom_list_array_crop=dicom_list_array[cy:cy+cm,cx:cx+cn,:]
					frangi_list_array_crop=frangi_list_array[cy:cy+cm,cx:cx+cn,:]
					vessel_list_array_crop=vessel_list_array[cy:cy+cm,cx:cx+cn,:] #
					print(dicom_list_array_crop.shape)
					self.image_slider.setMinimum(0)
					self.image_slider.setMaximum(dicom_list_array.shape[2]-1)
					self.state.setText('Done')

		# # dicom_list.reverse()
		# nor_dicom=self.NormlizDcm(np.transpose(dicom_list[0][cy:cy+cm,cx:cx+cn]))

		# height,width,bytesPerComponent=nor_dicom.shape
		# bytesPerLine=3*width
		# cvtColor(nor_dicom,COLOR_BGR2RGB,nor_dicom)

		# QImg=QImage(nor_dicom.data,width,height,bytesPerLine,QImage.Format_RGB888)
		# pixmap=QPixmap.fromImage(QImg)
		# self.OG_IMAGE.setPixmap(pixmap)
		# self.OG_IMAGE.setScaledContents(True)

		# self.SEG_IMAGE.setPixmap(pixmap)
		# self.SEG_IMAGE.setScaledContents(True)

	def ch_slice(self,value):

		try:
			dicom_list_array
		except NameError:
			pass
		else:
			try:
				pred_mask
			except NameError:
				slice_num=self.image_slider.value()
				nor_dicom=self.NormlizDcm(dicom_list_array[cy:cy+cm,cx:cx+cn,slice_num])
				img_over=nor_dicom.copy()

				img_over[:,:,0][vessel_list_array_crop[:,:,slice_num]>0]=0
				img_over[:,:,1][vessel_list_array_crop[:,:,slice_num]>0]=255
				img_over[:,:,2][vessel_list_array_crop[:,:,slice_num]>0]=0


				# print('slice size ',nor_dicom.shape)
				height,width,bytesPerComponent=nor_dicom.shape
				bytesPerLine=3*width

				cvtColor(nor_dicom,COLOR_BGR2RGB,nor_dicom)

				QImg=QImage(nor_dicom.data, width, height, bytesPerLine, QImage.Format_RGB888)
				pixmap=QPixmap.fromImage(QImg)
				self.OG_IMAGE.setPixmap(pixmap)
				self.OG_IMAGE.setScaledContents(True)

				QImg_seg=QImage(img_over.data, width, height, bytesPerLine, QImage.Format_RGB888) #
				pixmap_seg=QPixmap.fromImage(QImg_seg)
				self.MASK_IMAGE.setPixmap(pixmap_seg)
				self.MASK_IMAGE.setScaledContents(True)

				numb_of_slic='slice : '+str(slice_num)
				self.state.setText(numb_of_slic)
			else:
				slice_num=self.image_slider.value()
				nor_dicom=self.NormlizDcm(dicom_list_array[cy:cy+cm,cx:cx+cn,slice_num])
				# print('slice size ',nor_dicom.shape)
				height,width,bytesPerComponent=nor_dicom.shape
				bytesPerLine=3*width

				cvtColor(nor_dicom,COLOR_BGR2RGB,nor_dicom)

				img_over=nor_dicom.copy()

				img_over[:,:,0][vessel_list_array_crop[:,:,slice_num]>0]=0
				img_over[:,:,1][vessel_list_array_crop[:,:,slice_num]>0]=255
				img_over[:,:,2][vessel_list_array_crop[:,:,slice_num]>0]=0

				img_over[:,:,0][pred_mask[:,:,slice_num]>0.2]=255
				img_over[:,:,1][pred_mask[:,:,slice_num]>0.2]=0
				img_over[:,:,2][pred_mask[:,:,slice_num]>0.2]=0

				QImg=QImage(nor_dicom.data, width, height, bytesPerLine, QImage.Format_RGB888)
				pixmap=QPixmap.fromImage(QImg)
				self.OG_IMAGE.setPixmap(pixmap)
				self.OG_IMAGE.setScaledContents(True)

				QImg_seg=QImage(img_over.data, width, height, bytesPerLine, QImage.Format_RGB888)
				pixmap_seg=QPixmap.fromImage(QImg_seg)
				self.MASK_IMAGE.setPixmap(pixmap_seg)
				self.MASK_IMAGE.setScaledContents(True)

				numb_of_slic='slice : '+str(slice_num)
				self.state.setText(numb_of_slic)

	def predmask_fun(self,TDM_input,frangi=None):
		Ftype=self.frangi_type()

		if Ftype=='NO_FRANGI':
			root=Tk()
			root.withdraw()
			weight_path=askopenfilename(filetypes = (("h5 files","*.h5"),("all files","*.*")))

			self.state.setText('processing')
			TDM_input=TDM_input.astype(np.float32)
			m,n,z=TDM_input.shape

			if z%32!=0:
				z_p=z+(32-(z%32))
			else:
				z_p=z

			TDM=np.zeros((m,n,z_p),np.float32)

			TDM[:,:,0:z]=TDM_input[:,:,:]
			print(TDM.shape)
		
			TDM = (TDM - TDM.min()) / (TDM.max() - TDM.min())
		
			predmask = np.zeros(TDM.shape)
			self.model_NF.load_weights(weight_path)
			toto=int(((TDM.shape[0]-32)/32)+1)*int(((TDM.shape[1]-32)/32)+1)*int(((TDM.shape[2]-32)/32)+1)
			count=0
			for i in range(0,int(((TDM.shape[0]-32)/32)+1)):
				for j in range(0,int(((TDM.shape[1]-32)/32)+1)):
					for k in range(0,int(((TDM.shape[2]-32)/32)+1)):
						count+=1
						prog=(count/toto)*100
						self.progressBar.setValue(prog)
						x_1 , x_2 = 0 + i*32 , 32 + i*32
						y_1 , y_2 = 0 + j*32 , 32 + j*32
						z_1 , z_2 = 0 + k*32 , 32  + k*32

						volume = TDM[x_1:x_2,y_1:y_2,z_1:z_2]
						volume = np.reshape(volume, (1,) + volume.shape + (1,))

						pred   = np.argmax(self.model_NF.predict(volume), axis=-1)

						predmask[x_1:x_2,y_1:y_2,z_1:z_2] = predmask[x_1:x_2,y_1:y_2,z_1:z_2] + pred[0,:,:,:]

			mask_out=(predmask-np.min(predmask))/(np.max(predmask)-np.min(predmask))

			# self.progressBar.setValue(i)
			self.state.setText('done')
			print('done')
			return mask_out[:,:,0:z]

		elif Ftype=='USE_FRANGI':
			root=Tk()
			root.withdraw()
			weight_path=askopenfilename(filetypes = (("h5 files","*.h5"),("all files","*.*")))
 
			self.state.setText('processing')
			TDM_input=TDM_input.astype(np.float32)
			m,n,z=TDM_input.shape

			if z%32!=0:
				z_p=z+(32-(z%32))
			else:
				z_p=z

			TDM=np.zeros((m,n,z_p),np.float32)
			fran_=np.zeros((m,n,z_p),np.float32)

			TDM[:,:,0:z]=TDM_input[:,:,:]
			fran_[:,:,0:z]=frangi[:,:,:]
			print(TDM.shape)

			fran_[fran_>0.01]=1

			TDM[TDM>600]=600
			TDM[TDM<-200]=-200
			TDM = (TDM - TDM.min()) / (TDM.max() - TDM.min())

			# predmask = np.zeros(TDM.shape)
			self.model_F.load_weights(weight_path)
			predmask = np.zeros(TDM.shape)
			# temp = np.zeros(TDM.shape)
			count=0
			toto=int(((TDM.shape[0]-32)/32)+1)*int(((TDM.shape[1]-32)/32)+1)*int(((TDM.shape[2]-32)/32)+1)

			vessel_sk=skeletonize_3d(fran_)>0
			zn=z_p/4

			pointindex=np.argwhere(vessel_sk==1)
			center_pts=list()

			tmp_x=pointindex[0,0]
			tmp_y=pointindex[0,1]
			tmp_z=pointindex[0,2]

			temp_xp=0
			temp_yp=0
			temp_zp=0

			print('over lapping process')
			self.state.setText('over lapping process')

			for point in range(1,len(pointindex)):

				point_x=pointindex[point,0]
				point_y=pointindex[point,1]
				point_z=pointindex[point,2]

				dx=point_x-tmp_x
				dy=point_y-tmp_y
				dz=point_z-tmp_z

				dist=sqrt(dx**2+dy**2+dz**2)
				if dist>(32/4):
					center_pts.append([point_x,point_y,point_z])
					tmp_x=point_x
					tmp_y=point_y
					tmp_z=point_z

			print('volume predicting')
			self.state.setText('volume predicting')
			prog_count=0
			for center_point in center_pts:
				prog=(prog_count/len(center_pts))*100
				self.progressBar.setValue(prog)

				x_point=center_point[0]
				y_point=center_point[1]
				z_point=center_point[2]

				if x_point-16<0:
					temp_xp=x_point+16
				elif x_point+16>m:
					temp_xp=x_point-16
				else:
					temp_xp=x_point

				if y_point-16<0:
					temp_yp=y_point+16
				elif y_point+16>n:
					temp_yp=y_point-16
				else:
					temp_yp=y_point

				if z_point-16<0:
					temp_zp=z_point+16
				elif z_point+16>z_p:
					temp_zp=z_point-16
				else:
					temp_zp=z_point

				volume=TDM[temp_xp-16:temp_xp+16,temp_yp-16:temp_yp+16,temp_zp-16:temp_zp+16]
				volume=np.reshape(volume, (1,) + volume.shape + (1,))

				fran_=frangi[temp_xp-16:temp_xp+16,temp_yp-16:temp_yp+16,temp_zp-16:temp_zp+16]
				fran_= np.reshape(fran_, (1,) + fran_.shape + (1,))

				pred=np.argmax(self.model_F.predict([volume,fran_]), axis=-1)
				# pred[pred<0.5]=0

				for i in range(32):
					for j in range(32):
						for k in range(32):
							if pred[0,i,j,k]>predmask[temp_xp-16+i,temp_yp-16+j,temp_zp-16+k]:
								predmask[temp_xp-16+i,temp_yp-16+j,temp_zp-16+k]=pred[0,i,j,k]

				# predmask[temp_xp-16:temp_xp+16,temp_yp-16:temp_yp+16,temp_zp-16:temp_zp+16] = predmask[temp_xp-16:temp_xp+16,temp_yp-16:temp_yp+16,temp_zp-16:temp_zp+16] + pred[0,:,:,:]

			# predmask[predmask>1]=1
			mask_out_crop=(predmask[:,:,0:z]>0.5)*1

			dice=self.Dice(vessel_list_array_crop,mask_out_crop)
			print(dice)

			dice_str='Dice : '+str(dice)
			self.DICE.setText(dice_str)

			self.state.setText('done')
			print('done')

			return mask_out_crop

	def save_result(self):
		try:
			pred_mask
		except NameError:
			pass
		else:
			root=Tk()
			root.withdraw()
			save_path=asksaveasfilename(initialdir = "D:/",title = "Select file",filetypes = (("mat files","*.mat"),("all files","*.*")))
			print(save_path)
			savemat(save_path,{'pred_mask':pred_mask})

	def render_with_ground(self):
		try:
			pred_mask
		except NameError:
			# print('name error')
			pass
		else:
			# print('render_with_ground')
			vessel_list_array_crop[vessel_list_array_crop>0]=100
			pred_mask[pred_mask>0]=50
			pred_mask_p=pred_mask+vessel_list_array_crop
			data_matrix=pred_mask_p.astype(np.uint8)
			m,n,z=data_matrix.shape
			# data_matrix[data_matrix>0.5]=50
			vtk.vtkObject.GlobalWarningDisplayOff()

			dataImporter = vtk.vtkImageImport()

			data_string = data_matrix.tostring()
			dataImporter.CopyImportVoidPointer(data_string, len(data_string))

			dataImporter.SetDataScalarTypeToUnsignedChar()
			dataImporter.SetNumberOfScalarComponents(1)

			dataImporter.SetDataExtent(0, z-1, 0, n-1, 0, m-1)
			dataImporter.SetWholeExtent(0, z-1, 0, n-1, 0, m-1)

			alphaChannelFunc = vtk.vtkPiecewiseFunction()
			alphaChannelFunc.AddPoint(0, 0.0)
			alphaChannelFunc.AddPoint(50, 2)
			alphaChannelFunc.AddPoint(100, 2)
			alphaChannelFunc.AddPoint(150, 10)

			colorFunc = vtk.vtkColorTransferFunction()
			colorFunc.AddRGBPoint(50, 1.0, 0.0, 0.0)
			colorFunc.AddRGBPoint(100, 0.0, 1.0, 0.0)
			colorFunc.AddRGBPoint(150, 0.0, 1.0, 1.0)
			colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
			# colorFunc.AddRGBPoint(150, 0.0, 0.0, 1.0)

			volumeProperty = vtk.vtkVolumeProperty()
			volumeProperty.SetColor(colorFunc)
			volumeProperty.SetScalarOpacity(alphaChannelFunc)

			compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()

			volumeMapper = vtk.vtkVolumeRayCastMapper()
			volumeMapper.SetVolumeRayCastFunction(compositeFunction)
			volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

			volume = vtk.vtkVolume()
			volume.SetMapper(volumeMapper)
			volume.SetProperty(volumeProperty)

			renderer = vtk.vtkRenderer()
			renderWin = vtk.vtkRenderWindow()
			renderWin.AddRenderer(renderer)
			renderInteractor = vtk.vtkRenderWindowInteractor()
			renderInteractor.SetRenderWindow(renderWin)

			renderer.AddVolume(volume)

			renderer.SetBackground(1, 1, 1)
			renderWin.SetSize(400, 400)

			def exitCheck(obj, event):
			    if obj.GetEventPending() != 0:
			        obj.SetAbortRender(1)

			renderWin.AddObserver("AbortCheckEvent", exitCheck)

			renderInteractor.Initialize()
			renderWin.Render()
			renderWin.SetWindowName('3D Vessal')
			renderInteractor.Start()

	def render(self):

		try:
			pred_mask
		except NameError:
			pass
		else:
			pred_mask[pred_mask>0.5]=50
			data_matrix=pred_mask.astype(np.uint8)
			m,n,z=data_matrix.shape
			# data_matrix[data_matrix>0.5]=50
			vtk.vtkObject.GlobalWarningDisplayOff()

			dataImporter = vtk.vtkImageImport()

			data_string = data_matrix.tostring()
			dataImporter.CopyImportVoidPointer(data_string, len(data_string))

			dataImporter.SetDataScalarTypeToUnsignedChar()
			dataImporter.SetNumberOfScalarComponents(1)

			dataImporter.SetDataExtent(0, z-1, 0, n-1, 0, m-1)
			dataImporter.SetWholeExtent(0, z-1, 0, n-1, 0, m-1)

			alphaChannelFunc = vtk.vtkPiecewiseFunction()
			alphaChannelFunc.AddPoint(0, 0.0)
			alphaChannelFunc.AddPoint(50, 2)

			colorFunc = vtk.vtkColorTransferFunction()
			colorFunc.AddRGBPoint(50, 1.0, 0.0, 0.0)
			colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
			# colorFunc.AddRGBPoint(150, 0.0, 0.0, 1.0)

			volumeProperty = vtk.vtkVolumeProperty()
			volumeProperty.SetColor(colorFunc)
			volumeProperty.SetScalarOpacity(alphaChannelFunc)

			compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()

			volumeMapper = vtk.vtkVolumeRayCastMapper()
			volumeMapper.SetVolumeRayCastFunction(compositeFunction)
			volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

			volume = vtk.vtkVolume()
			volume.SetMapper(volumeMapper)
			volume.SetProperty(volumeProperty)

			renderer = vtk.vtkRenderer()
			renderWin = vtk.vtkRenderWindow()
			renderWin.AddRenderer(renderer)
			renderInteractor = vtk.vtkRenderWindowInteractor()
			renderInteractor.SetRenderWindow(renderWin)

			renderer.AddVolume(volume)

			renderer.SetBackground(1, 1, 1)
			renderWin.SetSize(400, 400)

			def exitCheck(obj, event):
			    if obj.GetEventPending() != 0:
			        obj.SetAbortRender(1)

			renderWin.AddObserver("AbortCheckEvent", exitCheck)

			renderInteractor.Initialize()
			renderWin.Render()
			renderWin.SetWindowName('3D Vessal')
			renderInteractor.Start()

	def testing(self):
		Ftype=self.frangi_type()
		global pred_mask
		if Ftype=='NO_FRANGI':
			try:
				dicom_list_array
			except NameError:
				pass
			else:
				pred_mask=self.predmask_fun(dicom_list_array_crop)
		elif Ftype=='USE_FRANGI':
			try:
				dicom_list_array
			except NameError:
				pass
			else:
				pred_mask=self.predmask_fun(dicom_list_array_crop,frangi=frangi_list_array_crop)


	def CTProjection(self,CTSet,Ptype='MIP'):

		m,n,z=CTSet.shape
		Output=np.zeros((m,n))

		if Ptype=='meanIP':
			for i in range(0,m):
				for j in range(0,n):
					Output[i,j]=CTSet[i,j,:].sum()/z
			return Output

		elif Ptype=='MIP':
			for i in range(0,m):
				for j in range(0,n):
					Output[i,j]=CTSet[i,j,:].max()

			print(Output.shape)
			return Output

	def dicom_Crop(self,dicom):
		# namedWindow('Crop Image', flags=WINDOW_NORMAL | WINDOW_FREERATIO)
		# proj_dicom=self.CTProjection(dicom)
		# proj_dicom=self.NormlizDcm(proj_dicom)
		# imshow('Crop Image',proj_dicom)
		# showCrosshair = True
		# fromCenter = False
		# rect = selectROI('Crop Image', proj_dicom, showCrosshair, fromCenter)
		# (x, y, w, h) = rect

		# waitKey(0)
		# destroyAllWindows()

		# imCrop = proj_dicom[y : y+h, x:x+w]
		# m,n,_=imCrop.shape
		
		# if m%32!=0:
		# 	m=m+(32-(m%32))
		# else:
		# 	m=m
		# if n%32!=0:
		# 	n=n+(32-(n%32))
		# else:
		# 	n=n

		x,y=0,0

		m,n,_=dicom.shape

		print(m,n)

		return x,y,m,n

	def restart_program(self):
		python = sys.executable
		execl(python, python, * sys.argv)

	def close(self):
		sys.modules[__name__].__dict__.clear()

if __name__=='__main__':
	app=QtWidgets.QApplication(sys.argv)
	MainWindow=QtWidgets.QMainWindow()
	ui=VnetWindow(MainWindow)
	MainWindow.setWindowTitle('Vnet CTA Segmentation Tool Ver1')

	MainWindow.show()
	sys.exit(app.exec_())