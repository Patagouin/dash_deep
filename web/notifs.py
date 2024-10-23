import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def SendMessage():
	msgFromAddress = "tonygaboulaud@patagouin.com"
	msgToAddress = "tonygaboulaud34@gmail.com"
	# on crée un e-mail
	message = MIMEMultipart("alternative")
	# on ajoute un sujet
	message["Subject"] = "[Trade] Buy"
	# un émetteur
	message["From"] = msgFromAddress
	# un destinataire
	message["To"] = msgToAddress
	# on crée un texte et sa version HTML
	#texte = f'''
	#Action CAP.PA
	#Achat 102.20
	#Quantité 23
	#Total {23*102.20}
	#'''

	html = f'''
	<html>
	<body>
	<h1>Action CAP.PA</h1>
	<p>Achat 102.20r</p>
	<b>Quantité 23</b>
	<br>
	<a href="https://datascientest.com">Total {23*102.20}</a>
	</body>
	</html>
	'''

	# on crée deux éléments MIMEText 
	#texte_mime = MIMEText(texte, 'plain')
	html_mime = MIMEText(html, 'html')

	# on attache ces deux éléments 
	#message.attach(texte_mime)
	message.attach(html_mime)

	# on crée la connexion
	context = ssl.create_default_context()
	with smtplib.SMTP_SSL("smtp.ionos.fr", 465, context=context) as server:
	  # connexion au compte
	  server.login(msgFromAddress, "Ravipitesse23$")
	  # envoi du mail
	  server.sendmail(msgFromAddress, msgToAddress, message.as_string())

