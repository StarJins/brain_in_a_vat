// std::io 모듈은 입출력 관련 기능을 제공합니다.
// `self`를 함께 가져오면 io 모듈 전체에 접근할 수 있고, Write 트레이트는 출력 스트림에 데이터를 기록하는 기능을 제공합니다.
use std::io::{self, Write};

// std::net 모듈의 TcpStream 구조체를 사용하여 TCP 연결을 생성 및 관리할 수 있습니다.
use std::net::TcpStream;

// 메인 함수는 I/O 연산이 실패할 수 있으므로 Result<(), std::io::Error>를 반환합니다.
// Rust에서 main 함수가 Result를 반환하면, 에러 발생 시 자동으로 에러를 출력하고 프로그램을 종료합니다.
fn main() -> std::io::Result<()> {
    // 사용자로부터 ID를 입력받기 위한 준비:
    // String::new()를 사용해 빈 문자열을 생성합니다.
    let mut id = String::new();
    
    // print! 매크로를 사용하여 사용자에게 입력을 요청합니다.
    // print!는 출력 후 자동 개행을 하지 않으므로, 사용자 입력 전에 화면에 바로 출력됩니다.
    print!("사용자 ID를 입력하세요: ");
    
    // flush 메서드는 stdout 버퍼에 남아있는 내용을 즉시 출력하도록 합니다.
    // '?' 연산자를 사용해 flush 과정에서 에러 발생 시 즉시 반환합니다.
    io::stdout().flush()?;
    
    // stdin().read_line(&mut id)는 표준 입력에서 한 줄을 읽어 id 변수에 저장합니다.
    // 이때 입력된 내용의 끝에 개행 문자('\n')가 포함됩니다.
    io::stdin().read_line(&mut id)?;
    
    // 입력 받은 문자열에서 앞뒤 공백 및 개행 문자를 제거합니다.
    let id = id.trim(); // trim()은 원본 데이터를 변경하지 않고 새로운 슬라이스(&str)를 반환합니다.

    // 서버에 연결하기:
    // TcpStream::connect는 지정된 주소("127.0.0.1:7878")로 TCP 연결을 시도하며,
    // 연결 성공 시 TcpStream 인스턴스를 반환합니다.
    let mut stream = TcpStream::connect("127.0.0.1:7878")?;
    println!("서버에 연결되었습니다.");

    // 서버로 먼저 사용자 ID를 전송합니다.
    // id.to_string()은 &str 타입인 id를 String으로 변환하며, "+ "\n""을 통해 메시지 끝에 개행 문자를 추가합니다.
    // as_bytes()를 사용해 문자열을 바이트 배열로 변환하여 네트워크로 전송할 수 있게 합니다.
    stream.write_all((id.to_string() + "\n").as_bytes())?;

    // 무한 루프를 사용해 사용자가 메시지를 입력할 때마다 서버로 전송합니다.
    loop {
        // 사용자 입력을 받을 빈 문자열을 생성합니다.
        let mut input = String::new();
        
        // 사용자에게 메시지 입력 요청을 출력합니다.
        print!("보낼 메시지를 입력하세요 (종료하려면 exit 입력): ");
        io::stdout().flush()?;
        
        // 표준 입력에서 한 줄을 읽어 input 변수에 저장합니다.
        io::stdin().read_line(&mut input)?;
        
        // 입력된 문자열에서 앞뒤 공백 및 개행 문자를 제거하여 실제 메시지로 사용합니다.
        let message = input.trim();

        // 사용자가 "exit"를 입력하면 대소문자를 구분하지 않고 클라이언트를 종료합니다.
        if message.eq_ignore_ascii_case("exit") {
            println!("클라이언트를 종료합니다.");
            break;
        }

        // 메시지 전송:
        // 메시지에 개행 문자를 추가한 후, String을 바이트 배열로 변환해 서버로 전송합니다.
        stream.write_all((message.to_string() + "\n").as_bytes())?;
        println!("메시지를 전송했습니다: {}", message);
    }

    // 모든 작업이 성공적으로 완료되었음을 나타내기 위해 Ok(())를 반환합니다.
    Ok(())
}
