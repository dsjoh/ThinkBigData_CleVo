package com.thinkbigdata.clevo.dto;

import com.thinkbigdata.clevo.role.Role;
import com.thinkbigdata.clevo.topic.TopicName;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
@Getter @Setter @Builder
public class UserDto {
    @NotBlank(message = "이메일 필드 오류")
    private String email;
    private String name;
    @NotBlank(message = "닉네임 필드 오류")
    private String nickName;
    private LocalDate birth;
    private String gender;
    @Min(value = 1) @Max(value = 10)
    private Integer level;
    @Min(value = 1) @Max(value = 10)
    private Integer target;
    private Role role;
    private String imgPath;
    private LocalDateTime createdDate;
    private LocalDateTime lastLoginDate;
    @Size(min = 3, max = 10)
    private List<TopicName> topic;
}
