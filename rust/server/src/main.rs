// 표준 입출력 라이브러리에서 BufRead 트레이트와 BufReader 구조체를 가져옵니다.
// BufRead 트레이트는 버퍼를 사용해 줄 단위로 읽는 기능을 제공하며,
// BufReader는 스트림에 버퍼링을 적용하여 I/O 성능을 개선합니다.
use std::io::{BufRead, BufReader};

// 표준 네트워킹 라이브러리에서 TcpListener를 가져옵니다.
// TcpListener는 지정된 주소와 포트에 바인딩하여 클라이언트의 TCP 연결 요청을 수신합니다.
use std::net::TcpListener;

// 표준 스레딩 라이브러리에서 thread 모듈을 가져옵니다.
// thread::spawn 함수를 통해 별도의 스레드에서 코드를 실행하여 동시성을 구현할 수 있습니다.
use std::thread;

// 메인 함수는 I/O 연산이 실패할 수 있으므로 Result<(), std::io::Error>를 반환합니다.
// Rust에서 main 함수가 Result를 반환하면, 에러 발생 시 자동으로 에러를 출력하고 프로그램을 종료합니다.
fn main() -> std::io::Result<()> {
    // TcpListener::bind 함수는 지정된 IP와 포트("127.0.0.1:7878")에 바인딩합니다.
    // 이때 '?' 연산자를 사용해 에러 발생 시 main 함수에서 즉시 반환하도록 처리합니다.
    let listener = TcpListener::bind("127.0.0.1:7878")?;
    println!("서버가 127.0.0.1:7878에서 대기 중입니다.");

    // listener.incoming()은 TcpStream의 결과(Result<TcpStream, Error>)를 반환하는 반복자(iterator)를 생성합니다.
    // 이 반복자를 사용해 들어오는 각 연결을 순차적으로 처리합니다.
    for stream in listener.incoming() {
        // match 문을 통해 연결 성공과 실패를 구분합니다.
        match stream {
            Ok(stream) => {
                // thread::spawn을 사용해 새로운 스레드를 생성합니다.
                // move 클로저를 사용하여 클로저 내부로 stream의 소유권을 이동시킵니다.
                thread::spawn(move || {
                    // 클라이언트 연결을 처리하는 handle_client 함수를 호출합니다.
                    handle_client(stream);
                });
            }
            // 연결 실패 시 에러 메시지를 표준 에러 출력(eprintln!)합니다.
            Err(e) => eprintln!("연결 오류: {}", e),
        }
    }
    // 모든 처리가 정상적으로 종료되었음을 나타내기 위해 Ok(())를 반환합니다.
    Ok(())
}

// 클라이언트의 연결을 처리하는 함수입니다.
// 매개변수 stream은 std::net::TcpStream 타입으로, 네트워크 연결을 나타내며,
// 이는 std::net 모듈의 TcpStream 구조체로 제공됩니다.
fn handle_client(stream: std::net::TcpStream) {
    // BufReader를 사용해 stream을 감싸면, std::io 모듈의 BufRead 트레이트를 활용하여
    // 버퍼링된 입출력을 수행할 수 있습니다. 이를 통해 작은 읽기 요청을 줄이고 성능을 개선합니다.
    let mut reader = BufReader::new(stream);
    
    // 클라이언트로부터 첫 번째 줄(예: 클라이언트 ID)을 저장하기 위한 빈 String 변수입니다.
    let mut id_line = String::new();

    // read_line은 BufRead 트레이트의 메서드로, 스트림에서 한 줄씩 읽어들입니다.
    // 반환 타입은 Result<usize, std::io::Error>이며, 읽은 바이트 수를 Ok로 반환합니다.
    // 조기 에러 처리를 위해 match를 사용하여 결과를 바로 처리합니다.
    let bytes_read = match reader.read_line(&mut id_line) {
        Ok(n) => n,  // 성공 시 읽은 바이트 수를 n에 저장
        Err(e) => {
            // 에러가 발생하면 표준 에러 출력(eprintln!)을 사용하여 오류 메시지를 출력한 후 함수 종료
            eprintln!("클라이언트 ID 읽기 오류: {}", e);
            return;
        }
    };

    // 읽은 바이트가 0이면, 이는 스트림의 끝(EOF)으로 클라이언트가 연결을 종료한 것으로 간주합니다.
    if bytes_read == 0 {
        println!("클라이언트가 연결을 종료했습니다.");
        return;
    }

    // 클라이언트 ID에 해당하는 id_line에서 불필요한 공백과 개행 문자를 제거한 후 String으로 변환합니다.
    let client_id = id_line.trim_end().to_string();
    println!("client {} 접속", client_id);

    // 이후 클라이언트로부터 전송되는 메시지를 읽기 위해 사용할 빈 버퍼입니다.
    let mut buffer = String::new();
    
    // 무한 루프를 통해 클라이언트가 연결을 유지하는 동안 계속해서 메시지를 읽습니다.
    loop {
        // 이전에 읽은 데이터를 제거하기 위해 buffer를 비웁니다.
        buffer.clear();

        // 클라이언트로부터 한 줄씩 읽어들입니다.
        let bytes_read = match reader.read_line(&mut buffer) {
            Ok(n) => n, // 성공 시 읽은 바이트 수를 n에 저장
            Err(e) => {
                // 읽기 중 에러 발생 시 에러 메시지를 출력한 후 루프를 종료합니다.
                eprintln!("client {} 메시지 읽기 오류: {}", client_id, e);
                break;
            }
        };

        // 읽은 바이트가 0이면 클라이언트가 연결을 종료한 것으로 간주하고 종료합니다.
        if bytes_read == 0 {
            println!("client {} 연결 종료", client_id);
            break;
        }

        // 읽은 메시지에서 개행 문자를 제거한 후 클라이언트 ID와 함께 출력합니다.
        println!("client {}: {}", client_id, buffer.trim_end());
    }
}
