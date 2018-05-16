import os
import xml.etree.ElementTree as ET

for index in range(0, 1601):
  
  nZeroes = 6 - len(str(index))
  fileName = '0'*nZeroes + str(index)

  try: 
    filename = '/home/deepak/Desktop/Deep/Deep-Learning/Ass1/Dog/Annotation2/' + fileName + '.xml'
    '''
    print filename
    if( os.path.isfile(filename) ):
        print 'Yo'
        print filename
        print 'Yo'
    ''' 
    if( index==0 ):
        print filename
    tree = ET.parse(filename)
    objs = tree.findall('object')
    for obj in objs:
        bbox = obj.find('bndbox')
        cls_ind = obj.find('name').text.lower().strip()
        
        if (index==0): 
            print 'Yo'
        #if( cls_ind!='dog' ):
        #  print 'yo', index

        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)

  except Exception:
    print index  
    pass    