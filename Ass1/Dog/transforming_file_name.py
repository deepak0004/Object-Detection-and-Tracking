import os

pp = 0
for i in range(0, 656):
    try:
      op = str(i)
      nZeroes = 6 - len(str(1700+pp))
      fileName = '0'*nZeroes + str(1700+pp)
      st1 = '/home/deepak/Desktop/Dog/Images/' + op + '.jpg'
      st2 = '/home/deepak/Desktop/Dog/Images2/'+ fileName + '.jpg'
      st3 = '/home/deepak/Desktop/Dog/Annotation/' + op + '.xml'
      st4 = '/home/deepak/Desktop/Dog/Annotation2/'+ fileName + '.xml'
      x1 = 'Images/' + op + '.jpg'
      x3 = 'Annotation/' + op + '.xml'
      if( os.path.isfile(x1) and os.path.isfile(x3) ):
          print st1
          os.rename(st1, st2) 
          os.rename(st3, st4) 
          pp += 1
    except Exception:
      pass	