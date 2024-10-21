const int ledPinRed = 13;
const int ledPinGreen=3;
char incomingByte;

void setup() {
  Serial.begin(9600);
  pinMode(ledPinRed, OUTPUT); // Configura o pino do LED como saída
  pinMode(ledPinGreen, OUTPUT); // Configura o pino do LED como saída
}

void loop() {
  while (!Serial.available());
    incomingByte = Serial.read();
    if (incomingByte == '0') {
    digitalWrite(ledPinRed, HIGH);
    digitalWrite(ledPinGreen, LOW); 
  } else if (incomingByte == '1') {
    digitalWrite(ledPinRed, LOW); 
    digitalWrite(ledPinGreen, HIGH); 
  }

}
