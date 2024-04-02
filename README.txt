=======================================================================================================================
==================== Roberto Falcone ================== BLS-GSM ======================= Tests results =================
=======================================================================================================================

All'interno dell'archivio "Tests.rar" sono presenti tutti i test effettuati:
	• image_log_[x] -> test effettuati sulle immagini con rumore gaussiano aggiunto con varianza [x]
	• image_log_real -> test effettuati sulle immagini reali fornite, già rumorose
	• image_log_sp -> test effettuati sulle immagini con aggiunta di rumore sale e pepe
		- 1 -> Intensità 0.05
		- 2 -> Intensità 0.1

Sono presenti anche due test non menzionati all'interno del report (image_log_0.05 e image_log_0.3), 
ossia test con immagini con rumore guassiano aggiunto a varianza 0.05 e 0.3.
	
All'interno di ogni cartella sono presenti le immagini elaborate ([x]rec.bmp) dall'algoritmo BLS-GSM, 
le immagini rumorose ([x]noisy.bmp) e le immagini originali ([x]origin.bmp) assieme a tutti i file txt contenenti 
gli indici di qualità per ogni metodo di denoising utilizzato (log.txt contiene quelle di BLS-GSM).
In ordine, i file log sono organizzati come segue: ["Numero dell'immagine"; "MSE"; "PSNR"; "SS"\n]