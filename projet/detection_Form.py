import cv2

#on va charger l'image
image = cv2.imread("pokemon.png")
edged = cv2.Canny(image, 30, 150)
while (1):
#afficher l'image
 cv2.imshow("Image", image)
 cv2.imshow("Edged", edged)
 if cv2.waitKey(1) & 0xff == ord('q'):
    break

cv2.destroyAllWindows()